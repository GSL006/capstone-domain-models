import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available!  Training on GPU...")
    print("GPU device name:", torch.cuda.get_device_name(0))  # Print GPU name
else:
    device = torch.device("cpu")
    print("CUDA is not available. Training on CPU...")

# Example tensor on the GPU
a = torch.tensor([1.0, 2.0]).to(device)
print(a)
