import torch
from models.mnist import mnist_model
from data.data_loader import MNIST_loader
from utils.training import train_model,evaluate,train_and_eval
from compression.pruning import prune_model,quantize_model
from utils.loading import load_csr_from_npz,save_model_npz,load_model_from_npz
from config import config_device

path = 'data' #path to the data folder
device = config_device()

train_loader,test_loader = MNIST_loader(path)
model = mnist_model()
model = model.to(device)

train_and_eval(model,train_loader,test_loader,device,epochs=1)
prune_model(model,0.95)
train_and_eval(model,train_loader,test_loader,device,epochs=1)

def count_mask_sparsity(model):
    total = 0
    zeros = 0
    for module in model.modules():
        if hasattr(module, "mask"):
            total += module.mask.numel()
            zeros += (module.mask == 0).sum().item()
    print(f"Masked sparsity: {100 * zeros / total:.2f}%")

count_mask_sparsity(model)
torch.save(model.state_dict(), "compressed_models/model.pth")
quantize_model(model,16)
train_and_eval(model,train_loader,test_loader,device,epochs=1)
npz_path = "compressed_models/compressed.npz"
save_model_npz(model,npz_path)

model2 = mnist_model()
model2 = load_model_from_npz(model,npz_path,device)
train_loader,test_loader = MNIST_loader(path)
model2 = model2.to(device)
print(evaluate(model2,test_loader,device))
print(model.classifier[0].__dict__.keys())
print(model.classifier[0].weight)