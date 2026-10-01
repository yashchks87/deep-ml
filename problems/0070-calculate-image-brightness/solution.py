import torch

def calculate_brightness(img) -> float:
    """
    Calculate the average brightness of a grayscale image using PyTorch.

    Args:
        img: A 2D list where each element represents a pixel value between 0-255.

    Returns:
        The average brightness rounded to two decimal places, or -1 for invalid inputs.
    """
    if not img or not img[0]:
        return -1
    
    w = len(img[0])
    if any(len(row) != w for row in img):
        return -1
    
    x = torch.tensor(img, dtype = torch.float32)

    if (x < 0).any() or (x > 255).any():
        return -1
    
    return round(x.mean().item(), 2)