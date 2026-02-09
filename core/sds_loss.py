import sys
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

sys.path.insert(0, '/mnt/workspace/fblan/trajectory-generation/video_sds_eval')


class SDSLossWrapper:
    """W-RFSDS loss for CHORD 4DGS using Wan 2.2."""

    def __init__(self, model_name, device='cuda:0', guidance_scale=6.0,
                 min_tau=0.02, max_tau=0.98, total_iterations=500,
                 target_h=480, target_w=832):
        self.model_name = model_name
        self.device_str = device
        self.guidance_scale = guidance_scale
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.total_iterations = total_iterations
        self.target_h = target_h
        self.target_w = target_w

        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self._loaded = False
        self._prompt_embeds_cache = {}

    def load_model(self):
        """Load the Wan 2.2 pipeline with all models on GPU."""
        from diffusers import WanPipeline
        from diffusers.schedulers import UniPCMultistepScheduler

        print(f"Loading Wan model: {self.model_name}")
        pipe = WanPipeline.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )

        # Fix text encoder weight tying
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
            te = pipe.text_encoder
            if hasattr(te, 'shared') and hasattr(te.encoder, 'embed_tokens'):
                sw = te.shared.weight
                ew = te.encoder.embed_tokens.weight
                if not (ew != 0).any() and (sw != 0).any():
                    te.encoder.embed_tokens.weight = te.shared.weight
                    print("Fixed text encoder weight tying")

        flow_shift = 8.0 if self.target_h <= 480 else 5.0
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=flow_shift
        )

        gpu_id = int(self.device_str.split(':')[-1]) if ':' in self.device_str else 0
        gpu_device = torch.device(f'cuda:{gpu_id}')

        # Place all models directly on GPU (enough VRAM available)
        self.vae = pipe.vae.to(dtype=torch.float32, device=gpu_device)
        self.vae.eval()

        self.transformer = pipe.transformer.to(device=gpu_device)
        self.transformer.eval()

        self.text_encoder = pipe.text_encoder.to(device=gpu_device)
        self.text_encoder.eval()

        self.tokenizer = pipe.tokenizer
        self._loaded = True
        self._gpu_device = gpu_device
        print(f"Wan model loaded (all models on {gpu_device})")

    def _get_device(self):
        return self._gpu_device

    def encode_prompt(self, prompt, max_sequence_length=226):
        if prompt in self._prompt_embeds_cache:
            return self._prompt_embeds_cache[prompt]

        # With CPU offload, text_encoder may be on CPU until forward is called
        # Use the GPU device and let the offload hook handle placement
        device = self._gpu_device
        text_inputs = self.tokenizer(
            [prompt], padding="max_length", max_length=max_sequence_length,
            truncation=True, add_special_tokens=True,
            return_attention_mask=True, return_tensors="pt",
        )
        ids = text_inputs.input_ids.to(device)
        mask = text_inputs.attention_mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()

        with torch.no_grad():
            embeds = self.text_encoder(ids, mask).last_hidden_state
            embeds = embeds.to(dtype=torch.bfloat16, device=device)

        embeds_list = [u[:v] for u, v in zip(embeds, seq_lens)]
        embeds = torch.stack([
            torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in embeds_list
        ], dim=0)

        self._prompt_embeds_cache[prompt] = embeds
        return embeds

    def get_tau_for_iteration(self, iteration, total_iterations=None):
        """Annealing: tau decreases from max to min."""
        if total_iterations is None:
            total_iterations = self.total_iterations
        target_cdf = 1.0 - iteration / (total_iterations + 1)
        target_cdf = np.clip(target_cdf, 0.001, 0.999)
        return float(np.sqrt(target_cdf * (self.max_tau**2 - self.min_tau**2) + self.min_tau**2))

    def compute_sds_loss(self, video: Tensor, text_prompt: str,
                         negative_prompt: str = "", iteration: int = None):
        """
        Compute W-RFSDS loss as a differentiable scalar.

        The trick: we create a loss = latents * sds_target.detach()
        whose gradient w.r.t. latents equals the SDS gradient direction.
        Since latents are computed from video through VAE (differentiable),
        this propagates gradients back to the video and the deformation params.

        video: [T, H, W, 3] float32 [0, 1] (from differentiable render)
        Returns: scalar loss, tau_value
        """
        if not self._loaded:
            raise RuntimeError("Call load_model() first")

        # video [T, H, W, 3] -> [1, 3, T, H, W] in [-1, 1]
        v = video.permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, T, H, W]
        v = v * 2.0 - 1.0

        # Resize to VAE resolution
        B, C, T, H, W = v.shape
        if H != self.target_h or W != self.target_w:
            v = F.interpolate(
                v.reshape(B, C * T, H, W),
                size=(self.target_h, self.target_w),
                mode='bilinear', align_corners=False
            ).reshape(B, C, T, self.target_h, self.target_w)

        # VAE encode (differentiable!)
        device = self._get_device()
        v_input = v.to(device=device, dtype=torch.float32)
        latent_dist = self.vae.encode(v_input)
        latents = latent_dist.latent_dist.mean  # use mean for differentiability (not sample)

        if hasattr(self.vae.config, 'latents_mean'):
            z_dim = getattr(self.vae.config, 'z_dim', latents.shape[1])
            lm = torch.tensor(self.vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(latents)
            ls = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(latents)
            latents = (latents - lm) * ls

        latents_bf16 = latents.to(dtype=torch.bfloat16)

        # Get tau
        tau_value = self.get_tau_for_iteration(iteration) if iteration is not None \
            else np.random.uniform(self.min_tau, self.max_tau)

        tau_expanded = torch.tensor([tau_value], device=device, dtype=torch.bfloat16).view(1, 1, 1, 1, 1)

        # Noise
        noise = torch.randn_like(latents_bf16)

        # RF interpolation: z_tau = (1-tau)*z + tau*eps
        noisy_latents = (1 - tau_expanded) * latents_bf16.detach() + tau_expanded * noise

        # Timestep
        t = torch.tensor([tau_value * 1000.0], device=device, dtype=torch.bfloat16)

        # Encode prompts
        prompt_embeds = self.encode_prompt(text_prompt).to(device)
        neg_prompt_embeds = self.encode_prompt(negative_prompt).to(device)

        # CFG forward
        with torch.no_grad():
            v_text = self.transformer(
                hidden_states=noisy_latents, timestep=t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False
            )[0]
            v_uncond = self.transformer(
                hidden_states=noisy_latents, timestep=t,
                encoder_hidden_states=neg_prompt_embeds,
                return_dict=False
            )[0]

        velocity_pred = v_uncond + self.guidance_scale * (v_text - v_uncond)

        # W-RFSDS target: v_pred - eps + z
        sds_target = (velocity_pred - noise + latents_bf16.detach()).detach()

        # SDS loss: gradient of this w.r.t. latents = sds_target
        # L = latents * sds_target (dot product, gradient = sds_target)
        # Note: the scalar loss value can be negative (it's a pseudo-loss for
        # gradient injection, not a real objective). What matters is the gradient
        # direction, not the loss value itself.
        loss = (latents.to(torch.bfloat16) * sds_target).sum() / latents.numel()

        # Diagnostics (all detached, no extra graph cost)
        with torch.no_grad():
            sds_metrics = {
                # SDS gradient direction magnitude (what gets injected)
                'sds/target_norm': sds_target.norm().item(),
                # Latent statistics
                'sds/latent_norm': latents_bf16.norm().item(),
                'sds/latent_std': latents_bf16.std().item(),
                # Noise vs prediction
                'sds/noise_norm': noise.norm().item(),
                'sds/vpred_norm': velocity_pred.norm().item(),
                # CFG effect: how much the text guidance changes the prediction
                'sds/cfg_delta_norm': (self.guidance_scale * (v_text - v_uncond)).norm().item(),
                # Cosine similarity between SDS target and latent (alignment)
                'sds/target_latent_cos': F.cosine_similarity(
                    sds_target.flatten().unsqueeze(0).float(),
                    latents_bf16.flatten().unsqueeze(0).float()
                ).item(),
                # Per-element mean of SDS target (sign shows push direction)
                'sds/target_mean': sds_target.mean().item(),
            }

        return loss, tau_value, sds_metrics
