
import sys
import torch
import numpy as np
import pandas as pd
import sklearn
import transformers
import sentence_transformers

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Transformers:", transformers.__version__)
print("Sentence Transformers:", sentence_transformers.__version__)

print("\nCUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:",
          round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
          "GB")

    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()

    print("GPU computation: SUCCESS")
    print("Tensor device:", y.device)
