import numpy as np

def get_color_features(image, mask):
    pixels = image[mask == 255]
    if pixels.size == 0:
        return [0]*6

    R, G, B = pixels[:,0], pixels[:,1], pixels[:,2]
    return [
        np.mean(R), np.mean(G), np.mean(B),
        np.std(R), np.std(G), np.std(B)
    ]
