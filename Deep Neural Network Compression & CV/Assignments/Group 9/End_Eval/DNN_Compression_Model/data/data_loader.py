import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import kagglehub
from tqdm import tqdm

def get_lbp_image(gray):
    lbp = np.zeros_like(gray, dtype=np.uint8)
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    for idx, (dy, dx) in enumerate(neighbors):
        shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
        val = (shifted >= gray).astype(np.uint8)
        lbp |= (val<<(7-idx))
    return lbp

def create_fruit_mask(bgr_img):
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    white_mask = cv2.inRange(hsv_img, lower_white, upper_white)
    fruit_mask = cv2.bitwise_not(white_mask)
    kernel = np.ones((5, 5), np.uint8)
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel)
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN, kernel)
    return fruit_mask

def extract_shape_features_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.zeros(8, dtype=np.float32)
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = w / h if h != 0 else 0
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
    moments = cv2.moments(c)
    hu_moments = cv2.HuMoments(moments).flatten()
    phi_1 = -1 * np.copysign(1.0, hu_moments[0]) * np.log10(np.abs(hu_moments[0])) if hu_moments[0] != 0 else 0
    phi_2 = -1 * np.copysign(1.0, hu_moments[1]) * np.log10(np.abs(hu_moments[1])) if hu_moments[1] != 0 else 0
    return np.array([area, perimeter, w, h, aspect_ratio, circularity, phi_1, phi_2], dtype=np.float32)

def extract_color_features(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mean, std_dev = cv2.meanStdDev(hsv_img, mask=mask)
    return mean[0][0], std_dev[0][0], mean[1][0], std_dev[1][0], mean[2][0], std_dev[2][0]

class FruitsCVDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')])[:65]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img in os.listdir(cls_dir):
                if img.lower().endswith(('.png', '.jpeg', '.jpg')):
                    self.images.append((os.path.join(cls_dir, img), self.class_to_idx[cls]))
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        
        bgr_img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        
        lbp_img = get_lbp_image(gray_img)
        canny_img = cv2.Canny(gray_img, 100, 200)
        mask = create_fruit_mask(bgr_img)
        
        color_features = extract_color_features(bgr_img)
        shape_features = extract_shape_features_from_mask(mask)
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        lbp_resized = cv2.resize(lbp_img, (64, 64))
        canny_resized = cv2.resize(canny_img, (64, 64))
        
        lbp = torch.tensor(lbp_resized, dtype=torch.float32).unsqueeze(0) / 255.0
        canny = torch.tensor(canny_resized, dtype=torch.float32).unsqueeze(0) / 255.0
        shape = torch.tensor(shape_features, dtype=torch.float32)
        color = torch.tensor(color_features, dtype=torch.float32)
        label = torch.tensor(label)

        return image, lbp, canny, shape, color, label

class FastFruitsDataset(Dataset):
    def __init__(self, tensor_data):
        self.img = tensor_data['img']
        self.lbp = tensor_data['lbp']
        self.canny = tensor_data['canny']
        self.shape = tensor_data['shape']
        self.color = tensor_data['color']
        self.label = tensor_data['label']
        
    def __len__(self):
        return len(self.label)
        
    def __getitem__(self, idx):
        return self.img[idx], self.lbp[idx], self.canny[idx], self.shape[idx], self.color[idx], self.label[idx]

def process_and_cache(dataset, cache_path, num_workers=4):
    print(f"Extracting CV Features and building cache: {cache_path}")
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=num_workers)
    
    all_imgs, all_lbps, all_cannys, all_shapes, all_colors, all_labels = [], [], [], [], [], []
    
    for img, lbp, canny, shape, color, label in tqdm(loader, desc="Processing Images"):
        all_imgs.append(img)
        all_lbps.append(lbp)
        all_cannys.append(canny)
        all_shapes.append(shape)
        all_colors.append(color)
        all_labels.append(label)
        
    tensor_data = {
        'img': torch.cat(all_imgs),
        'lbp': torch.cat(all_lbps),
        'canny': torch.cat(all_cannys),
        'shape': torch.cat(all_shapes),
        'color': torch.cat(all_colors),
        'label': torch.cat(all_labels)
    }
    torch.save(tensor_data, cache_path)
    return tensor_data

def get_fruits_loaders(batch_size=32):
    path = kagglehub.dataset_download("moltean/fruits")
    train_root = os.path.join(path, "fruits-360_100x100", "fruits-360", "Training")
    test_root = os.path.join(path, "fruits-360_100x100", "fruits-360", "Test")
    
    train_dataset_raw = FruitsCVDataset(train_root)
    test_dataset_raw = FruitsCVDataset(test_root)
    
    num_classes = len(train_dataset_raw.classes)
    
    cache_dir = "data_cache"
    os.makedirs(cache_dir, exist_ok=True)
    train_cache_path = os.path.join(cache_dir, "train_cache_65.pt")
    test_cache_path = os.path.join(cache_dir, "test_cache_65.pt")
    
    if os.path.exists(train_cache_path):
        print("Loading cached Train features...")
        train_data = torch.load(train_cache_path)
    else:
        train_data = process_and_cache(train_dataset_raw, train_cache_path)
        
    if os.path.exists(test_cache_path):
        print("Loading cached Test features...")
        test_data = torch.load(test_cache_path)
    else:
        test_data = process_and_cache(test_dataset_raw, test_cache_path)
        
    fast_train = FastFruitsDataset(train_data)
    fast_test = FastFruitsDataset(test_data)
    
    train_loader = DataLoader(fast_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(fast_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, test_loader, num_classes
