import torch
print(f"PyTorch version: {torch.__version__}")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("MPS backend is available.")
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print(f"Test tensor on MPS: {x}")
else:
    print("MPS backend is not available.")
