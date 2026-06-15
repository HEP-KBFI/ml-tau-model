import torch
import torch.nn as nn
from mltau.models.MLPMixer import MLPMixerBackbone

class MixerTau(nn.Module):
    """
    MixerTau model for tau-tagging tasks.
    Uses an MLP-Mixer backbone instead of Particle Transformer.
    """
    def __init__(
        self,
        input_dim: int,
        task: str,
        n_constituents: int = 20,
        num_dm_classes: int = 6,
        embed_dim: int = 128,
        head_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.task = task
        self.embed_dim = embed_dim
        
        # Backbone
        self.backbone = MLPMixerBackbone(
            n_constituents=n_constituents,
            n_features=input_dim,
            embed_dim=embed_dim
        )
        
        # Heads (Reusing the same logic as in SingleParTau.py)
        head_hidden = embed_dim // 2
        if self.task == "decay_mode":
            self.classification_head = nn.Sequential(
                nn.Linear(embed_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, num_dm_classes),
            )
        elif self.task == "kinematics":
            self.regression_head = nn.Sequential(
                nn.Linear(embed_dim, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, 5),
            )
        elif self.task == "is_tau":
            self.binary_head = nn.Sequential(
                nn.Linear(embed_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(head_dropout / 2),
                nn.Linear(head_hidden, 2),
            )
        elif self.task == "charge":
            self.binary_head = nn.Sequential(
                nn.Linear(embed_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, 1),
            )
        else:
            raise NotImplementedError(
                f"This model is not suitable for the chosen task of {self.task}"
            )

    def forward(
        self,
        cand_features,
        cand_kinematics_pxpypze=None,
        cand_mask=None,
        return_embedding=False,
    ):
        # cand_features: (N, C, P)
        # cand_mask: (N, 1, P)
        
        # Backbone pass
        x_cls = self.backbone(cand_features, mask=cand_mask) # (N, embed_dim)
        
        # Head pass
        if self.task == "decay_mode":
            output = (self.classification_head(x_cls),)
        elif self.task == "kinematics":
            output = (self.regression_head(x_cls),)
        elif self.task == "is_tau":
            output = (self.binary_head(x_cls),)
        elif self.task == "charge":
            output = (self.binary_head(x_cls).squeeze(-1),)
        else:
            raise NotImplementedError(
                f"This model is not suitable for the chosen task of {self.task}"
            )

        if return_embedding:
            return output, x_cls
        return output
