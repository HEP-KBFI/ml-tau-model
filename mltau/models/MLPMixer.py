import torch
import torch.nn as nn

class MLPMixerBackbone(nn.Module):
    """
    MLP-Mixer backbone inspired by the HGQ jet classifier.
    
    This model performs feature-mixing and token-mixing on particle constituents.
    """
    def __init__(self, n_constituents: int, n_features: int, embed_dim: int = 16):
        super().__init__()
        self.n_constituents = n_constituents
        self.n_features = n_features
        self.embed_dim = embed_dim
        
        # Initial feature mixing: (N, P, C) -> (N, P, 16) -> (N, P, C)
        # Note: The original model uses fixed size 16 for internal layers
        self.feature_mixing = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, n_features),
            nn.ReLU()
        )
        
        # Token mixing: (N, P, C) -> (N, C, P) -> (N, C, P) -> (N, P, C)
        # Operates across the particle dimension
        self.token_mixing = nn.Sequential(
            nn.Linear(n_constituents, n_constituents),
            nn.ReLU()
        )
        
        # Post-mixing stage
        self.stage2 = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        
        # Learned pooling: (N, P, 16) -> (N, 16, P) -> (N, 16, 1) -> (N, 16)
        # Equivalent to Dense(1) on the P dimension
        self.pooling = nn.Sequential(
            nn.Linear(n_constituents, 1),
            nn.ReLU()
        )
        
        # Final embedding refinement
        self.final_mlp = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, embed_dim) # Output embed_dim
        )

    def forward(self, x, mask=None):
        # x: (N, C, P) - follow SingleParTau convention
        x = x.transpose(1, 2) # (N, P, C)
        
        # Feature mixing
        residual = x
        x = self.feature_mixing(x)
        
        # Token mixing
        x = x.transpose(1, 2) # (N, C, P)
        x = self.token_mixing(x)
        x = x.transpose(1, 2) # (N, P, C)
        
        # First residual connection
        x = x + residual
        
        # Second stage
        x = self.stage2(x) # (N, P, 16)
        
        # Learned pooling
        x = x.transpose(1, 2) # (N, 16, P)
        x = self.pooling(x) # (N, 16, 1)
        x = x.squeeze(-1) # (N, 16)
        
        # Final MLP
        x = self.final_mlp(x) # (N, embed_dim)
        
        return x
