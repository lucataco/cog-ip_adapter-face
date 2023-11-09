# Prediction interface for Cog
from cog import BasePredictor, Input, Path
import os
import sys
sys.path.extend(['/IP-Adapter'])
import torch
import shutil
from PIL import Image
from typing import List
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from ip_adapter import IPAdapterPlus

base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "/IP-Adapter/models/image_encoder/"
ip_ckpt = "/IP-Adapter/models/ip-adapter-plus-face_sd15.bin"
device = "cuda"
MODEL_CACHE = "model-cache"
VAE_CACHE = "vae-cache"

def load_image(path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        vae = AutoencoderKL.from_pretrained(
            vae_model_path,
            cache_dir=VAE_CACHE
        ).to(dtype=torch.float16)
        # load SD pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None,
            cache_dir=MODEL_CACHE
        )

    def predict(
        self,
        image: Path = Input(
             description="Input face image",
             default=None
        ),
        prompt: str = Input(
            description="Prompt",
            default="photo of a beautiful girl wearing casual shirt in a garden"
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        image = Image.open(image)
        image.resize((256, 256))

        ip_model = IPAdapterPlus(self.pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

        images = ip_model.generate(
            pil_image=image,
            num_samples=num_outputs,
            num_inference_steps=num_inference_steps,
            seed=seed,
            prompt=prompt
        )

        output_paths = []
        for i, _ in enumerate(images):
            output_path = f"/tmp/out-{i}.png"
            images[i].save(output_path)
            output_paths.append(Path(output_path))
            
        return output_paths
