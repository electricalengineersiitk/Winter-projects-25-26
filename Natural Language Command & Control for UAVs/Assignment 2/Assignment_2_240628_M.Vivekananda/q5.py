import torch
x=torch.tensor(4.0,requires_grad=True)
y=x**3+2*x
y.backward()
pytorch_grad=x.grad.item()
print(f"Pytorch Gradient when taking at x=4 : {pytorch_grad}")
manual_grad=3*(4.0**2)+2
if not torch.isclose(torch.tensor(float(pytorch_grad)), torch.tensor(float(manual_grad))):
    raise ValueError(f"Did not match PyTorch: {pytorch_grad}, Manual: {manual_grad}")
else:
    print("Passed ...gradients aree matched.")