
from diffusers import StableDiffusionPipeline
import torch
import logging
import gc
import io

class ImageService:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
        self.model_id = model_id
        self.device = device
        self.pipeline = None
        self.logger = logging.getLogger("ImageService")

    def load_pipeline(self):
        """
        Loads the SD pipeline only when needed to save VRAM.
        """
        if self.pipeline:
            return

        self.logger.info(f"Loading Stable Diffusion ({self.model_id})...")
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                safety_checker=None, # Save VRAM
                requires_safety_checker=False 
            )
            # Efficient Offloading: Keeps model in RAM, moves to GPU only for inference
            # This is perfect for "Low VRAM" scenarios (~3GB peak usage)
            self.pipeline.enable_model_cpu_offload() 
            
            # Alternative: pipeline.to("cuda") <--- This would eat 4GB+ constantly. Don't use.
            
            self.logger.info("Stable Diffusion loaded with CPU Offload enabled.")
        except Exception as e:
            self.logger.error(f"Failed to load Stable Diffusion: {e}")
            self.pipeline = None

    def unload_pipeline(self):
        """
        Aggressively unloads the pipeline to free up VRAM/RAM.
        """
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
            self.logger.info("Stable Diffusion unloaded.")

    async def generate_image(self, prompt, negative_prompt="ugly, blurry, low quality"):
        """
        Generates an image and returns bytes.
        """
        self.load_pipeline()
        
        if not self.pipeline:
            return None

        self.logger.info(f"Generating image for: {prompt}")
        try:
            # Run inference
            # 30 steps is a good balance for SD1.5
            image = self.pipeline(
                prompt=prompt, 
                negative_prompt=negative_prompt, 
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]
            
            # Save to BytesIO
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Optional: Unload immediately if user is strictly VRAM constrained?
            # User said "ideally under 3GB". CPU Offload keeps VRAM near 0 when idle.
            # But RAM usage might be high (2-3GB).
            # I'll enable a config option for "aggressive_unload" later if needed.
            # For now, let's keep it loaded in RAM (via offload) for faster subsequent gens.
            
            return img_byte_arr
            
        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            return None
