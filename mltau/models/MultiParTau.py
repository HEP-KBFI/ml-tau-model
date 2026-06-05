import contextlib
import copy
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
        # Dropout applied to shared-token heads (tagging, charge, DM) to counter
        # overtraining.  Kinematics head is intentionally left unregularized since
        # it is already the slowest learner.
        head_dropout: float = 0.1,
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
        # Two CLS tokens: one shared by tagging/charge/DM, one dedicated to kinematics.
        # This costs 2x (not 4x) in cls_blocks while still letting the DM head
        # read kinematic context (e.g. invariant mass) via the kinematics token.
        self.cls_token_shared = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_kinematics = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token_shared, std=0.02)
        trunc_normal_(self.cls_token_kinematics, std=0.02)

        # Dedicated cls_blocks and norm for the kinematics pathway.  Copied from
        # the inherited (shared) cls_blocks so the architecture is identical, but
        # weights are independent so classification gradients cannot corrupt the
        # kinematics attention layers.
        self.cls_blocks_kinematics = copy.deepcopy(self.cls_blocks)
        self.norm_kinematics = copy.deepcopy(self.norm)

        # Classification head for decay mode classification.
        # Takes the DM CLS embedding concatenated with the (detached) kinematics
        # CLS embedding so the DM head can exploit kinematic context without
        # sending classification gradients back into the kinematics CLS token.
        # 512 → 128 → 6: the compression layer helps separate the 6 DM classes
        # whose boundaries are non-trivial (e.g. DM 1 vs 2 differ by one π⁰).
        # Charge and tagging are left as single linear heads — they are simple
        # binary decisions well-represented in a 256-dim embedding and deeper
        # heads would only add overfit risk.
        dm_hidden = embed_dim // 2
        self.classification_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim + embed_dim, dm_hidden),
            nn.GELU(),
            nn.Linear(dm_hidden, num_dm_classes),
        )
        # Regression head: no dropout — kinematics is the hardest task and
        # the slowest learner, so we do not regularize it.
        hidden_dim = embed_dim // 2
        self.regression_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 5),
        )
        # Tagging is the second hardest task so use half the dropout rate to
        # avoid over-regularizing it relative to charge.
        # Two-class head: background = class 0, signal = class 1.
        self.tau_id_head = nn.Sequential(
            nn.Dropout(head_dropout / 2),
            nn.Linear(embed_dim, 2),
        )
        self.tau_charge_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, 1),
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token_shared", "cls_token_kinematics"}

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
            # Keep a snapshot of the raw embedding before the shared backbone
            # transforms it.  The kinematics cls_blocks will attend to this instead
            # of the backbone output, so the regression pathway is never exposed to
            # the classification-biased representation learned by self.blocks.
            cand_features_embed_raw = cand_features_embed  # same storage, no copy

            attn_mask = None
            if cand_kinematics_pxpypze is not None and self.pair_embed is not None:
                attn_mask = self.pair_embed(cand_kinematics_pxpypze).view(
                    -1, num_particles, num_particles
                )  # (N*num_heads, P, P)

            # transform particles (shared backbone — classification pathway only)
            for block in self.blocks:
                cand_features_embed = block(
                    cand_features_embed,
                    x_cls=None,
                    padding_mask=padding_mask,
                    attn_mask=attn_mask,
                )

            # transform per-jet class tokens
            N = cand_features_embed.size(1)

            # Shared pathway: tagging, charge, DM — attends to backbone output
            x_cls_shared = self.cls_token_shared.expand(1, N, -1)  # (1, N, C)
            for block in self.cls_blocks:
                x_cls_shared = block(
                    cand_features_embed, x_cls=x_cls_shared, padding_mask=padding_mask
                )
            x_shared = self.norm(x_cls_shared.squeeze(0))  # (N, C)

            # Kinematics pathway: attends to the PRE-backbone raw embedding so it
            # is never exposed to the classification-biased backbone representation.
            # The pair_embed attention bias (geometry) is passed through unchanged
            # so the kinematics blocks still see relative angles/distances.
            x_cls_kin = self.cls_token_kinematics.expand(1, N, -1)  # (1, N, C)
            for block in self.cls_blocks_kinematics:
                x_cls_kin = block(
                    cand_features_embed_raw, x_cls=x_cls_kin, padding_mask=padding_mask
                )
            x_kinematics = self.norm_kinematics(x_cls_kin.squeeze(0))  # (N, C)

            # As fc_params is an empty list, then basically we have been using one Linear layer only.
            # Now introduce the different heads also here.
            # Output raw logits - activations will be applied by loss functions or during inference

            output = {
                "is_tau": self.tau_id_head(
                    x_shared
                ),  # (N, 2) - raw logits (background=0, signal=1)
                "charge": self.tau_charge_head(x_shared).squeeze(
                    -1
                ),  # (N,) - raw logits
                # Detach x_kinematics so DM gradients do not corrupt the kinematics
                # regression token. The DM head still sees kinematic context
                # (e.g. invariant mass) but cannot steer the kinematics token.
                "decay_mode": self.classification_head(
                    torch.cat([x_shared, x_kinematics.detach()], dim=-1)
                ),  # (N, num_dm_classes) - raw logits
                "kinematics": self.regression_head(x_kinematics),  # (N, 5)
            }

            return output
