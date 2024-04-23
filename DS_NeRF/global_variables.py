import torch
# from diffusers import StableDiffusionInpaintPipeline

# Initialize the global GPU tensor
# image_inpainted = StableDiffusionPipelineOutput()
grad = torch.zeros((1, 4, 64, 64), device=torch.device("cuda"))