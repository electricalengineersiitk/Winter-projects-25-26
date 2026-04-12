import torch
from dataloader import get_loader

DATA_PATH = "/kaggle/input/fruits/fruits-360_100x100/fruits-360/Training"

loader = get_loader(DATA_PATH)

for batch in loader:
    img, lbp, canny, mask, shape_f, color_f, label = batch

    shape_f = torch.stack(shape_f).T
    color_f = torch.stack(color_f).T

    print("Image:", img.shape)
    print("LBP:", lbp.shape)
    print("Canny:", canny.shape)
    print("Shape features:", shape_f.shape)
    print("Color features:", color_f.shape)
    print("Label:", label.shape)
    break
