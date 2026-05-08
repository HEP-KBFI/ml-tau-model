import contextlib
import torch
import torch.nn as nn
from mltau.models.ParticleTransformer import ParticleTransformer, trunc_normal_


class ParTau(ParticleTransformer):
    def __init__(
        self,
        input_dim: int,
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

        # We will have a total of 4 heads: decay mode, kinematic, charge and tauID.

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim

        # Replace the single inherited cls_token with 4 per-task tokens.
        # Each token attends to the particle cloud independently through cls_blocks,
        # giving each head the same private representational capacity as a single-head model.
        del self.cls_token
        self.cls_token_tagging = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_charge = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_decay_mode = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_kinematics = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token_tagging, std=0.02)
        trunc_normal_(self.cls_token_charge, std=0.02)
        trunc_normal_(self.cls_token_decay_mode, std=0.02)
        trunc_normal_(self.cls_token_kinematics, std=0.02)

        # Classification head for decay mode classification.
        # Takes the DM CLS embedding concatenated with the (detached) kinematics
        # CLS embedding so the DM head can exploit kinematic context without
        # sending classification gradients back into the kinematics CLS token.
        self.classification_head = nn.Linear(embed_dim + embed_dim, num_dm_classes)
        # Regression head: small MLP for richer non-linear mapping from CLS to targets.
        # [log_pt, deta, delta_sin(phi), delta_cos(phi), log_m]
        hidden_dim = embed_dim // 2
        self.regression_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 5),
        )
        # Binary heads for tau-tagging and charge reco
        self.tau_id_head = nn.Linear(embed_dim, 1)
        self.tau_charge_head = nn.Linear(embed_dim, 1)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token_tagging",
            "cls_token_charge",
            "cls_token_decay_mode",
            "cls_token_kinematics",
        }

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
        amp_ctx = (
            torch.amp.autocast("cuda") if self.use_amp else contextlib.nullcontext()
        )
        with amp_ctx:
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
            # The cls_blocks are designed for a single CLS token per sample.
            # To give each task its own private CLS token, we tile the particle
            # features 4x along the batch dimension so each task slot runs as an
            # independent sample through the cls_blocks in one efficient pass.
            N = cand_features_embed.size(1)
            # (P, 4N, C) and (4N, P)
            feat_4x = cand_features_embed.repeat(1, 4, 1)
            mask_4x = padding_mask.repeat(4, 1)
            # Stack the 4 task tokens along the batch dim: (1, 4N, C)
            cls_tokens = torch.cat(
                [
                    self.cls_token_tagging.expand(1, N, -1),
                    self.cls_token_charge.expand(1, N, -1),
                    self.cls_token_decay_mode.expand(1, N, -1),
                    self.cls_token_kinematics.expand(1, N, -1),
                ],
                dim=1,
            )  # (1, 4N, C)
            for block in self.cls_blocks:
                cls_tokens = block(feat_4x, x_cls=cls_tokens, padding_mask=mask_4x)
            cls_tokens = self.norm(cls_tokens.squeeze(0))  # (4N, C)
            x_tagging = cls_tokens[:N]  # (N, C)
            x_charge = cls_tokens[N : 2 * N]  # (N, C)
            x_decay_mode = cls_tokens[2 * N : 3 * N]  # (N, C)
            x_kinematics = cls_tokens[3 * N :]  # (N, C)

            # As fc_params is an empty list, then basically we have been using one Linear layer only.
            # Now introduce the different heads also here.
            # Output raw logits - activations will be applied by loss functions or during inference

            output = {
                "is_tau": self.tau_id_head(x_tagging).squeeze(-1),  # (N,) - raw logits
                "charge": self.tau_charge_head(x_charge).squeeze(
                    -1
                ),  # (N,) - raw logits
                # Detach x_kinematics so that DM classification gradients cannot
                # flow back into the kinematics CLS token and corrupt its regression
                # representation.  The DM head still benefits from kinematic context
                # via the detached features, but kinematics training is unaffected.
                "decay_mode": self.classification_head(
                    torch.cat([x_decay_mode, x_kinematics.detach()], dim=-1)
                ),  # (N, num_dm_classes) - raw logits
                "kinematics": self.regression_head(
                    x_kinematics
                ),  # (N, 5) - [log_pt, deta, delta_sin(phi), delta_cos(phi), log_m]
            }

            return output
