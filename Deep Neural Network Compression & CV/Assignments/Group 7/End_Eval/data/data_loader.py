import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def get_lbp(gray):
    h, w   = gray.shape
    lbp    = np.zeros((h, w), dtype=np.uint8)
    center = gray[1:h-1, 1:w-1]
    lbp[1:h-1, 1:w-1] |= (gray[0:h-2, 0:w-2] >= center).astype(np.uint8) << 7
    lbp[1:h-1, 1:w-1] |= (gray[0:h-2, 1:w-1] >= center).astype(np.uint8) << 6
    lbp[1:h-1, 1:w-1] |= (gray[0:h-2, 2:w  ] >= center).astype(np.uint8) << 5
    lbp[1:h-1, 1:w-1] |= (gray[1:h-1, 2:w  ] >= center).astype(np.uint8) << 4
    lbp[1:h-1, 1:w-1] |= (gray[2:h,   2:w  ] >= center).astype(np.uint8) << 3
    lbp[1:h-1, 1:w-1] |= (gray[2:h,   1:w-1] >= center).astype(np.uint8) << 2
    lbp[1:h-1, 1:w-1] |= (gray[2:h,   0:w-2] >= center).astype(np.uint8) << 1
    lbp[1:h-1, 1:w-1] |= (gray[1:h-1, 0:w-2] >= center).astype(np.uint8) << 0
    return lbp


def get_canny(gray):
    return cv2.Canny(gray, 100, 200)


def extract_color_features(img):
    hsv     = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    return np.array([
        np.mean(h), np.std(h),
        np.mean(s), np.std(s),
        np.mean(v), np.std(v)
    ])


def extract_shape_features(gray):
    _, mask    = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.zeros(6)

    cnt        = max(contours, key=cv2.contourArea)
    area       = cv2.contourArea(cnt)
    perimeter  = cv2.arcLength(cnt, True)
    img_area   = gray.shape[0] * gray.shape[1]
    area_ratio = area / img_area

    x, y, w, h  = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h > 0 else 0

    hull      = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity  = area / hull_area if hull_area > 0 else 0

    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-10)

    moments = cv2.moments(cnt)
    hu      = cv2.HuMoments(moments).flatten()
    hu_1    = -np.sign(hu[0]) * np.log10(np.abs(hu[0]) + 1e-10)
    hu_2    = -np.sign(hu[1]) * np.log10(np.abs(hu[1]) + 1e-10)

    return np.array([area_ratio, aspect_ratio, solidity,
                     circularity, hu_1, hu_2])


class Fruit360_DataLoader(Dataset):
    def __init__(self, path):
        self.samples      = []
        self.class_to_idx = {}
        classes           = sorted(os.listdir(path))

        for idx, cls in enumerate(classes):
            if cls.startswith('.'):
                continue
            cls_path = os.path.join(path, cls)
            if not os.path.isdir(cls_path):
                continue
            self.class_to_idx[cls] = idx
            for img_name in os.listdir(cls_path):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                self.samples.append((os.path.join(cls_path, img_name), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"cv2.imread returned None for {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (100, 100))
        except Exception as e:
            print(f"[DataLoader] Skipping bad image {img_path}: {e}")
            return {
                "lbp"           : torch.zeros(1, 100, 100),
                "canny"         : torch.zeros(1, 100, 100),
                "color_features": torch.zeros(6),
                "shape_features": torch.zeros(6),
                "label"         : torch.tensor(label, dtype=torch.long)
            }

        gray           = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lbp            = get_lbp(gray)
        canny          = get_canny(gray)
        color_features = extract_color_features(img)
        shape_features = extract_shape_features(gray)

        return {
            "lbp"           : torch.tensor(lbp).unsqueeze(0).float() / 255.0,
            "canny"         : torch.tensor(canny).unsqueeze(0).float() / 255.0,
            "color_features": torch.tensor(color_features).float(),
            "shape_features": torch.tensor(shape_features).float(),
            "label"         : torch.tensor(label, dtype=torch.long)
        }


def data_loader(training, test):
    train_dataset = Fruit360_DataLoader(training)
    test_dataset  = Fruit360_DataLoader(test)

    train_loader = DataLoader(
        train_dataset,
        batch_size        = 64,
        shuffle           = True,
        num_workers       = 0,
        pin_memory        = False,
        persistent_workers= False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size        = 64,
        shuffle           = False,
        num_workers       = 0,
        pin_memory        = False,
        persistent_workers= False
    )
    return train_loader, test_loader