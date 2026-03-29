import os
import cv2
from torch.utils.data import Dataset

from features.mask import get_fruit_mask
from features.shape import get_shape_features
from features.color import get_color_features
from features.canny import get_canny_image
from features.lbp import get_lbp_image

class FruitDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        classes = sorted(os.listdir(root_dir))
        for idx, cls in enumerate(classes):
            cls_path = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            self.class_to_idx[cls] = idx
            for img in os.listdir(cls_path):
                self.image_paths.append(os.path.join(cls_path, img))
                self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = get_fruit_mask(image)
        shape_f = get_shape_features(mask)
        color_f = get_color_features(image, mask)
        canny = get_canny_image(image)
        lbp = get_lbp_image(image)

        return image, lbp, canny, mask, shape_f, color_f, label

