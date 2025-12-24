import torch
from torch.utils.data import Dataset
class NUmber_Dataaset(Dataset):
    def __init__(self,data_list):
        self.data=data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        number=self.data[idx]
        sample_input=torch.tensor(float(number))
        sample_target=torch.tensor(float(number*2))
        return sample_input,sample_target
    