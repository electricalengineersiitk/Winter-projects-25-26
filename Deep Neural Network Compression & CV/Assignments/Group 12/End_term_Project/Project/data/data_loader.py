

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (100, 100)


CANNY_LOW: int = 100
CANNY_HIGH: int = 200


LBP_NEIGHBOURS: int = 8


def read_image(path: str) -> np.ndarray:
    
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_image(
    img: np.ndarray,
    size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def to_gray(img_rgb: np.ndarray) -> np.ndarray:
    
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)



def compute_canny(
    gray: np.ndarray,
    low: int = CANNY_LOW,
    high: int = CANNY_HIGH,
) -> np.ndarray:
    
    return cv2.Canny(gray, low, high)


def compute_lbp(gray: np.ndarray) -> np.ndarray:
    
    h, w = gray.shape
    gray_f = gray.astype(np.float32)

    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0,  1),
        ( 1,  1), ( 1,  0), ( 1, -1),
        ( 0, -1),
    ]

    lbp = np.zeros((h, w), dtype=np.uint8)

    center = gray_f[1:-1, 1:-1]

    for bit, (dr, dc) in enumerate(offsets):
        r0, r1 = 1 + dr, h - 1 + dr      
        c0, c1 = 1 + dc, w - 1 + dc      
        neighbour = gray_f[r0:r1, c0:c1]
        lbp[1:-1, 1:-1] |= (neighbour >= center).astype(np.uint8) << bit

    return lbp


def extract_color_features(img_rgb: np.ndarray) -> np.ndarray:
   
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    return np.array(
        [
            np.mean(h_ch), np.mean(s_ch), np.mean(v_ch),
            np.std(h_ch),  np.std(s_ch),  np.std(v_ch),
        ],
        dtype=np.float32,
    )


def extract_shape_features(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return np.zeros(6, dtype=np.float32)

    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, closed=True)

    image_area = gray.shape[0] * gray.shape[1]
    area_ratio = area / image_area if image_area > 0 else 0.0

    x, y, bw, bh = cv2.boundingRect(cnt)
    aspect_ratio = bw / bh if bh > 0 else 0.0

    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidity = area / hull_area if hull_area > 0 else 0.0

    bb_area = bw * bh
    extent = area / bb_area if bb_area > 0 else 0.0

    circularity = (
        (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    )

    return np.array(
        [area_ratio, perimeter, aspect_ratio, solidity, extent, circularity],
        dtype=np.float32,
    )


class FruitsDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size

        self.image_paths: List[str] = []
        self.labels: List[int] = []
        self.class_to_idx: Dict[str, int] = {}

        self._build_index()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        classes = sorted(
            entry.name
            for entry in os.scandir(self.root_dir)
            if entry.is_dir() and not entry.name.startswith(".")
        )

        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for cls in classes:
            cls_dir = os.path.join(self.root_dir, cls)
            label = self.class_to_idx[cls]

            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(SUPPORTED_EXTENSIONS):
                    self.image_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(label)

    def _process(self, path: str) -> Dict:
        img = read_image(path)
        img = resize_image(img, self.image_size)
        gray = to_gray(img)

        lbp = compute_lbp(gray)
        canny = compute_canny(gray)

        color_feats = extract_color_features(img)
        shape_feats = extract_shape_features(gray)
        features = np.concatenate([color_feats, shape_feats])  # (12,)

        return {
            "image": img,
            "gray": gray,
            "lbp": lbp,
            "canny": canny,
            "features": features,
        }

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict:
        sample = self._process(self.image_paths[idx])
        sample["label"] = self.labels[idx]
        return sample
    
    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    @property
    def idx_to_class(self) -> Dict[int, str]:
        return {v: k for k, v in self.class_to_idx.items()}

    def __repr__(self) -> str:
        return (
            f"FruitsDataset("
            f"root='{self.root_dir}', "
            f"samples={len(self)}, "
            f"classes={self.num_classes}, "
            f"image_size={self.image_size})"
        )

def _collate_fn(batch: List[Dict]) -> Dict:
    keys = batch[0].keys()
    out: Dict = {}

    for key in keys:
        values = [sample[key] for sample in batch]

        if key == "label":
            out[key] = torch.tensor(values, dtype=torch.long)

        elif key == "features":
            # Already float32
            out[key] = torch.from_numpy(np.stack(values))

        else:
            # uint8 arrays → normalise to float32 in [0, 1]
            arr = np.stack(values).astype(np.float32) / 255.0

            if key == "image":
                # (B, H, W, C) → (B, C, H, W) for CNN compatibility
                arr = arr.transpose(0, 3, 1, 2)

            out[key] = torch.from_numpy(arr)

    return out


def get_dataloader(
    root_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> Tuple[DataLoader, FruitsDataset]:
    
    dataset = FruitsDataset(root_dir=root_dir, image_size=image_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
    )

    return loader, dataset

def MNIST_loader(path, batch_size=64):
    
    import os
    train_dir = os.path.join(path, "Training")
    test_dir  = os.path.join(path, "Test")

    if not _os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}\n"
            f"Expected Fruits-360 layout: {path}/Training  and  {path}/Test"
        )
    if not _os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}\n"
            f"Expected Fruits-360 layout: {path}/Training  and  {path}/Test"
        )
    

    train_loader, _ = get_dataloader(train_dir, batch_size=batch_size, shuffle=True)
    test_loader,  _ = get_dataloader(test_dir,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
