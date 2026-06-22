import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.model = nn.Sequential(
            nn.Embedding(vocabulary_size, 16),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer
        embeddings = self.model[0](x)
        avg = torch.mean(embeddings, dim=1)
        linear_out = self.model[1](avg)
        output = self.model[2](linear_out)
        # Return a B, 1 tensor and round to 4 decimal places
        return torch.round(output, decimals=4)
