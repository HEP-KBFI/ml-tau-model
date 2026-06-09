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

        # Four task-specific CLS tokens.
        self.cls_token_tau_id = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_charge = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_dm = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_kinematics = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token_tau_id, std=0.02)
        trunc_normal_(self.cls_token_charge, std=0.02)
        trunc_normal_(self.cls_token_dm, std=0.02)
        trunc_normal_(self.cls_token_kinematics, std=0.02)

        # Four separate CLS blocks (one per task) and their corresponding norms.
        # We reuse the inherited self.cls_blocks and self.norm as templates.
        self.cls_blocks_tau_id = copy.deepcopy(self.cls_blocks)
        self.cls_blocks_charge = copy.deepcopy(self.cls_blocks)
        self.cls_blocks_dm = copy.deepcopy(self.cls_blocks)
        self.cls_blocks_kinematics = copy.deepcopy(self.cls_blocks)

        self.norm_tau_id = copy.deepcopy(self.norm)
        self.norm_charge = copy.deepcopy(self.norm)
        self.norm_dm = copy.deepcopy(self.norm)
        self.norm_kinematics = copy.deepcopy(self.norm)

        # Clean up inherited shared blocks as we now use task-specific ones.
        del self.cls_blocks
        del self.norm

        # Simplified task-specific readout heads with a consistent 1-layer FFN architecture.
        # (Linear -> GELU -> Dropout -> Linear)
        head_hidden = embed_dim // 2

        self.tau_id_head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout / 2),
            nn.Linear(head_hidden, 2),
        )
        self.tau_charge_head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, num_dm_classes),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden),
            nn.GELU(),
            # No dropout for kinematics as it's the hardest learner.
            nn.Linear(head_hidden, 5),
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token_tau_id",
            "cls_token_charge",
            "cls_token_dm",
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

            # transform particles (shared backbone for all tasks)
            for block in self.blocks:
                cand_features_embed = block(
                    cand_features_embed,
                    x_cls=None,
                    padding_mask=padding_mask,
                    attn_mask=attn_mask,
                )

            # transform per-jet task tokens
            N = cand_features_embed.size(1)

            # Tau ID pathway
            x_cls_tau_id = self.cls_token_tau_id.expand(1, N, -1)
            for block in self.cls_blocks_tau_id:
                x_cls_tau_id = block(
                    cand_features_embed, x_cls=x_cls_tau_id, padding_mask=padding_mask
                )
            x_tau_id = self.norm_tau_id(x_cls_tau_id.squeeze(0))

            # Charge pathway
            x_cls_charge = self.cls_token_charge.expand(1, N, -1)
            for block in self.cls_blocks_charge:
                x_cls_charge = block(
                    cand_features_embed, x_cls=x_cls_charge, padding_mask=padding_mask
                )
            x_charge = self.norm_charge(x_cls_charge.squeeze(0))

            # Decay Mode pathway
            x_cls_dm = self.cls_token_dm.expand(1, N, -1)
            for block in self.cls_blocks_dm:
                x_cls_dm = block(
                    cand_features_embed, x_cls=x_cls_dm, padding_mask=padding_mask
                )
            x_dm = self.norm_dm(x_cls_dm.squeeze(0))

            # Kinematics pathway
            x_cls_kin = self.cls_token_kinematics.expand(1, N, -1)
            for block in self.cls_blocks_kinematics:
                x_cls_kin = block(
                    cand_features_embed, x_cls=x_cls_kin, padding_mask=padding_mask
                )
            x_kinematics = self.norm_kinematics(x_cls_kin.squeeze(0))

            # Output raw logits - activations will be applied by loss functions or during inference
            output = {
                "is_tau": self.tau_id_head(x_tau_id),
                "charge": self.tau_charge_head(x_charge).squeeze(-1),
                "decay_mode": self.classification_head(x_dm),
                "kinematics": self.regression_head(x_kinematics),
            }

            return output
