from torch.utils.data import DataLoader
from dataset.fruit_dataset import FruitDataset

def get_loader(path, batch_size=8):
    dataset = FruitDataset(path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader
