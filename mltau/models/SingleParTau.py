import torch
import torch.nn as nn
from mltau.models.ParticleTransformer import ParticleTransformer


class ParTau(ParticleTransformer):
    def __init__(
        self,
        input_dim: int,
        task: str,
        num_dm_classes=6,  # For decay mode classification head only
        # network configurations
        pair_input_dim: int = 4,
        pair_extra_dim: int = 0,
        remove_self_pair: bool = False,
        use_pre_activation_pair: bool = True,
        embed_dims: list[int] = [256, 512, 256],
        pair_embed_dims: list[int] = [64, 64, 64],
        num_heads: int = 8,
        num_layers: int = 8,
        num_cls_layers: int = 2,
        block_params=None,
        cls_block_params: dict = {
            "dropout": 0,
            "attn_dropout": 0,
            "activation_dropout": 0,
        },
        fc_params: list = [],
        activation: str = "gelu",
        # misc
        trim: bool = True,
        for_inference: bool = False,
        use_amp: bool = False,
        metric: str = "eta-phi",
        verbosity: int = 0,
        **kwargs,
    ):
        # Don't pass num_classes to parent since we implement our own heads
        super().__init__(
            input_dim=input_dim,
            num_classes=1,
            pair_input_dim=pair_input_dim,
            pair_extra_dim=pair_extra_dim,
            remove_self_pair=remove_self_pair,
            use_pre_activation_pair=use_pre_activation_pair,
            embed_dims=embed_dims,
            pair_embed_dims=pair_embed_dims,
            num_heads=num_heads,
            num_layers=num_layers,
            num_cls_layers=num_cls_layers,
            block_params=block_params,
            cls_block_params=cls_block_params,
            fc_params=fc_params,
            activation=activation,
            # misc
            trim=trim,
            for_inference=for_inference,
            use_amp=use_amp,
            metric=metric,
            verbosity=verbosity,
            **kwargs,
        )
        self.for_inference = for_inference
        self.use_amp = use_amp
        self.task = task

        # We will have a total of 4 heads: decay mode, kinematic, charge and tauID.

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        if self.task == "decay_mode":
            # Classification head for decay mode classification
            self.classification_head = nn.Linear(embed_dim, num_dm_classes)
        elif self.task == "kinematics":
            # Regression head: small MLP for richer non-linear mapping from CLS to targets.
            # [log(pt_gen/pt_reco), delta_eta, delta_sin(phi), delta_cos(phi), log(m_gen/m_reco)]
            hidden_dim = embed_dim // 2
            self.regression_head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 5),
            )
        elif self.task == "is_tau":
            # Binary head for tau-tagging
            self.binary_head = nn.Linear(embed_dim, 1)
        elif self.task == "charge":
            # Two-logit head to match the en-reg charge-classification setup
            self.binary_head = nn.Linear(embed_dim, 2)
        else:
            raise NotImplementedError(
                f"This model is not suitable for the chosen task of {self.task}"
            )

    def forward(
        self,
        cand_features,
        cand_kinematics_pxpypze=None,
        cand_mask=None,
    ):
        # cand_features: (N=num_batches, C=num_features, P=num_particles)
        # cand_kinematics_pxpypze: (N, 4, P) [px,py,pz,energy]
        # cand_mask: (N, 1, P) -- real particle = 1, padded = 0
        cand_mask = cand_mask.type(torch.bool)
        padding_mask = ~cand_mask.squeeze(1)  # (N, 1, P) -> (N, P)
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            num_particles = cand_features.size(-1)

            # input embedding
            cand_features_embed = self.embed(cand_features).masked_fill(
                ~cand_mask.permute(2, 0, 1), 0
            )  # (P, N, C)
            attn_mask = None
            if cand_kinematics_pxpypze is not None and self.pair_embed is not None:
                attn_mask = self.pair_embed(cand_kinematics_pxpypze).view(
                    -1, num_particles, num_particles
                )  # (N*num_heads, P, P)

            # transform particles
            for block in self.blocks:
                cand_features_embed = block(
                    cand_features_embed,
                    x_cls=None,
                    padding_mask=padding_mask,
                    attn_mask=attn_mask,
                )

            # transform per-jet class tokens
            cls_tokens = self.cls_token.expand(
                1, cand_features_embed.size(1), -1
            )  # (1, N, C)
            for block in self.cls_blocks:
                cls_tokens = block(
                    cand_features_embed, x_cls=cls_tokens, padding_mask=padding_mask
                )
            x_cls = self.norm(cls_tokens).squeeze(0)

            # As fc_params is an empty list, then basically we have been using one Linear layer only.
            # Now introduce the different heads also here.

            if self.task == "decay_mode":
                output = (
                    self.classification_head(x_cls),
                )  # (N, num_dm_classes) raw logits
            elif self.task == "kinematics":
                # Regression output: [log(pt_gen/pt_reco), delta_eta, delta_sin(phi), delta_cos(phi), log(m_gen/m_reco)]
                output = (self.regression_head(x_cls),)  # (N, 5)
            elif self.task == "is_tau":
                # Return logits; sigmoid is applied outside for logging and evaluation.
                output = (self.binary_head(x_cls).squeeze(-1),)  # (N,)
            elif self.task == "charge":
                # Two-class logits for charge classification
                output = (self.binary_head(x_cls),)  # (N, 2)
            else:
                raise NotImplementedError(
                    f"This model is not suitable for the chosen task of {self.task}"
                )

            return output
