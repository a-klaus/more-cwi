"""
Complex word classifier using embedding model followed by simple neural network.
"""

import torch
from torch import nn

# Set up the classifier network
class ComplexWordClassifier(nn.Module):
    """
    A simple network mapping from word embeddings to complex/simple labels.
    The embedding model is followed by a small neural network with one hidden
    layer, dropout and ReLU activation.

    Args:
        embedding_model (SentenceTransformer): a Sentence Transformer model
            to map input words to high-dimensional embeddings
        hidden_size (int): hidden layer size
        dropout_rate (float): dropout to apply after hidden layer
    """
    def __init__(self, embedding_model, hidden_size=400, dropout_rate=0.1):
        super().__init__()
        self.embedding_model = embedding_model
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(384, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        x = self.embedding_model.encode(x)  # Embed input word
        x = torch.from_numpy(x)
        logits = self.linear_relu_stack(x)  # Apply network
        return logits

