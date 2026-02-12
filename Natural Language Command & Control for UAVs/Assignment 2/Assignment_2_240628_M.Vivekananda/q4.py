import torch

def process_image(images):
    processed_tensor=torch.where(images<0.5,0.0,1.0)
    return processed_tensor


images = torch.rand(4,28,28)
masked_images = process_image(images)