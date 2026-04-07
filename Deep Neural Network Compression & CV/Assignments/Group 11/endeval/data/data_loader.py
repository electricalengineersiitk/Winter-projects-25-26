import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def compute_lbp(img_gray):
    h, w = img_gray.shape
    res = np.zeros((h, w), dtype=np.uint8)
    ref = img_gray[1:h-1, 1:w-1]
    res[1:h-1, 1:w-1] |= (img_gray[0:h-2, 0:w-2] >= ref).astype(np.uint8) << 7
    res[1:h-1, 1:w-1] |= (img_gray[0:h-2, 1:w-1] >= ref).astype(np.uint8) << 6
    res[1:h-1, 1:w-1] |= (img_gray[0:h-2, 2:w  ] >= ref).astype(np.uint8) << 5
    res[1:h-1, 1:w-1] |= (img_gray[1:h-1, 2:w  ] >= ref).astype(np.uint8) << 4
    res[1:h-1, 1:w-1] |= (img_gray[2:h,   2:w  ] >= ref).astype(np.uint8) << 3
    res[1:h-1, 1:w-1] |= (img_gray[2:h,   1:w-1] >= ref).astype(np.uint8) << 2
    res[1:h-1, 1:w-1] |= (img_gray[2:h,   0:w-2] >= ref).astype(np.uint8) << 1
    res[1:h-1, 1:w-1] |= (img_gray[1:h-1, 0:w-2] >= ref).astype(np.uint8) << 0
    return res

def edge_detect(img_gray):
    return cv2.Canny(img_gray, 100, 200)

def get_color_stats(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    return np.array([np.mean(h), np.std(h), np.mean(s), np.std(s), np.mean(v), np.std(v)])

def get_geometry(img_gray):
    _, m = cv2.threshold(img_gray, 245, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return np.zeros(6)
    c = max(cnts, key=cv2.contourArea)
    a, p = cv2.contourArea(c), cv2.arcLength(c, True)
    r_area = a / (img_gray.shape[0] * img_gray.shape[1])
    _, _, bw, bh = cv2.boundingRect(c)
    r_aspect = bw / bh if bh > 0 else 0
    h = cv2.convexHull(c)
    ha = cv2.contourArea(h)
    sol = a / ha if ha > 0 else 0
    circ = 4 * np.pi * a / (p ** 2 + 1e-10)
    mu = cv2.moments(c)
    hu = cv2.HuMoments(mu).flatten()
    h1 = -np.sign(hu[0]) * np.log10(np.abs(hu[0]) + 1e-10)
    h2 = -np.sign(hu[1]) * np.log10(np.abs(hu[1]) + 1e-10)
    return np.array([r_area, r_aspect, sol, circ, h1, h2])

class ImageSet(Dataset):
    def __init__(self, folder):
        self.items = []
        labels = sorted(os.listdir(folder))
        for i, name in enumerate(labels):
            if name.startswith('.'): continue
            p = os.path.join(folder, name)
            if not os.path.isdir(p): continue
            for f in os.listdir(p):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.items.append((os.path.join(p, f), i))

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        f_path, target = self.items[i]
        try:
            raw = cv2.imread(f_path)
            if raw is None: raise ValueError("Empty image")
            img = cv2.resize(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), (100, 100))
        except Exception:
            return {"lbp": torch.zeros(1, 100, 100), "canny": torch.zeros(1, 100, 100),
                    "color_features": torch.zeros(6), "shape_features": torch.zeros(6),
                    "label": torch.tensor(target, dtype=torch.long)}

        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        l_feat = torch.tensor(compute_lbp(g)).unsqueeze(0).float() / 255.0
        e_feat = torch.tensor(edge_detect(g)).unsqueeze(0).float() / 255.0
        c_feat = torch.tensor(get_color_stats(img)).float()
        g_feat = torch.tensor(get_geometry(g)).float()
        return {"lbp": l_feat, "canny": e_feat, "color_features": c_feat, "shape_features": g_feat,
                "label": torch.tensor(target, dtype=torch.long)}

def create_loaders(tr_dir, te_dir):
    ds_tr, ds_te = ImageSet(tr_dir), ImageSet(te_dir)
    l_tr = DataLoader(ds_tr, batch_size=64, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    l_te = DataLoader(ds_te, batch_size=64, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
    return l_tr, l_te