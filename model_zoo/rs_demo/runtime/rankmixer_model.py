"""RankMixer compute blocks, ported from the QuantaRec RankMixer model.

This module is self-contained (no quantarec / model_config dependency) so it can
run inside the RecStore rs_demo harness.  It faithfully reproduces the production
RankMixer architecture used by the quanta_rankmixer project:

  MaskBlock -> per-segment LT projection -> [TokenMixer + PFFN] x N -> mean -> PLE

Production hyper-parameters (config/features/params.json):
  tokens_split_dim = 2400, rankmixer_blocks = 2, gatenum = 6
  task groups: ctr(5) + time(7) + interact(11) + video(9) + poi(4) = 36 tasks
  mmoe_units [1024,1024] dice+bn_after, task_small_units [128], task_units [64,64]

The module replaces build_hybrid_dense_arch (DLRM) when --model rankmixer is
selected.  It consumes the sparse embedding lookups produced by either the
RecStore PS backend (bagpipe + prefetch) or the local dynamic-embedding backend,
so the compute path is identical across the two architectures being compared.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


# --------------------------------------------------------------------------
# Activation + initializer helpers (ported from model/activation.py, utils.py)
# --------------------------------------------------------------------------

class Dice(nn.Module):
    """Dice activation wrapping a BatchNorm1d (matches TF Dice semantics)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):
        super().__init__()
        self.bn = nn.BatchNorm1d(hidden_size, eps=eps, momentum=momentum,
                                 affine=affine, track_running_stats=track_running_stats)
        self.sigmoid = nn.Sigmoid()
        self.alphas = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x_p = self.bn(x)
        elif x.dim() == 3:
            x_p = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            shape = x.shape
            x_p = self.bn(x.reshape(-1, self.bn.num_features)).view(*shape)
        x_p = self.sigmoid(x_p)
        return self.alphas * (1.0 - x_p) * x + x_p * x


def get_activation(activation, activation_params: Optional[dict] = None):
    if activation is None:
        return nn.Identity()
    activation = str(activation).lower()
    activation_params = dict(activation_params or {})
    if activation == "dice":
        activation_params.setdefault("hidden_size", activation_params.get("hidden_size", 64))
        activation_params.update({"eps": 1e-09, "momentum": 0.01, "affine": False})
        return Dice(**activation_params)
    if activation in ("none", "identity", "null"):
        return nn.Identity()
    if activation == "gelu":
        return nn.GELU(**activation_params)
    if activation == "relu":
        return nn.ReLU()
    if activation in ("swish", "silu"):
        return nn.SiLU()
    if activation == "tanh":
        return nn.Tanh()
    if activation == "sigmoid":
        return nn.Sigmoid()
    if activation == "leaky_relu":
        return nn.LeakyReLU()
    return nn.Identity()


def _init_linear(layer: nn.Linear, init_type: str = "normal", std: float = 0.01) -> None:
    if init_type == "normal":
        nn.init.normal_(layer.weight, mean=0.0, std=std)
    elif init_type == "glorot":
        nn.init.xavier_uniform_(layer.weight)
    elif init_type == "const":
        nn.init.constant_(layer.weight, std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


# --------------------------------------------------------------------------
# TokenMixer (ported from model/tokenmixer.py)
# --------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, epsilon: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, 1, hidden_size))
        self.beta = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=-1, keepdim=True)
        variance = inputs.var(dim=-1, keepdim=True, unbiased=False)
        normed = (inputs - mean) / torch.sqrt(variance + self.epsilon)
        return self.gamma * normed + self.beta


class TokenMixer(nn.Module):
    """Token mixer: reshape/transpose mixing across the token dimension + residual LN."""

    def __init__(self, hidden_dim: int, num_tokens: int, epsilon: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.layer_norm = LayerNorm(hidden_dim, epsilon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        bs, _, hidden_dim = inputs.shape
        num_token = self.num_tokens
        assert hidden_dim == self.hidden_dim, (
            f"Input hidden_dim ({hidden_dim}) doesn't match module hidden_dim ({self.hidden_dim})"
        )
        inputs_ = inputs.reshape(bs, num_token, num_token, hidden_dim // num_token)
        inputs_ = inputs_.transpose(1, 2)
        inputs_ = inputs_.reshape(bs, num_token, hidden_dim)
        return self.layer_norm(inputs + inputs_)


# --------------------------------------------------------------------------
# PFFN (ported from model/pffn.py)
# --------------------------------------------------------------------------

class PerTokenLayerNorm(nn.Module):
    def __init__(self, token_num: int, token_dim: int, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(token_num, 1, token_dim))
        self.beta = nn.Parameter(torch.zeros(token_num, 1, token_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / (var + self.epsilon).sqrt()
        return self.gamma * x_norm + self.beta


class BatchLinear(nn.Module):
    def __init__(self, token_num: int, token_dim: int, hidden_dim: int, use_bias: bool):
        super().__init__()
        self.use_bias = use_bias
        self.fc_weight = nn.Parameter(torch.empty(token_num, token_dim, hidden_dim))
        nn.init.trunc_normal_(self.fc_weight, mean=0.0, std=0.01)
        if use_bias:
            self.fc_bias = nn.Parameter(torch.zeros(token_num, 1, hidden_dim))
        else:
            self.fc_bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.matmul(self.fc_weight)
        return x + self.fc_bias if self.fc_bias is not None else x


class PositionWiseFFN(nn.Module):
    def __init__(self, token_num: int, token_dim: int, hidden_size: int,
                 activation=None, activation_params: Optional[dict] = None,
                 use_bias: bool = True, scale_factor: float = 3):
        super().__init__()
        self.layers = nn.Sequential(
            BatchLinear(token_num, token_dim, hidden_size * scale_factor, use_bias),
            get_activation(activation, activation_params),
            BatchLinear(token_num, hidden_size * scale_factor, token_dim, use_bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PFFN(nn.Module):
    def __init__(self, token_num, token_dim, hidden_size, activation=None,
                 activation_params=None, enable_batch_ln=False, scale_factor=3,
                 use_adaptiveffn=False):
        super().__init__()
        self.use_adaptiveffn = use_adaptiveffn
        # AdaptiveFFN is not used in the production RankMixer config (scale=3,
        # relu activation); keep the simple PositionWiseFFN path.
        self.ffn = PositionWiseFFN(
            token_num, token_dim, hidden_size,
            activation=activation, activation_params=activation_params,
            use_bias=True, scale_factor=scale_factor,
        )
        if enable_batch_ln:
            self.ln = PerTokenLayerNorm(token_num, token_dim)
        else:
            self.ln = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        x = x.permute(1, 0, 2)          # (B, S, D) -> (S, B, D)
        x = self.ffn(x)
        if self.ln is not None:
            x = self.ln(x)
        x = x.permute(1, 0, 2)          # (S, B, D) -> (B, S, D)
        return x + x_in


# --------------------------------------------------------------------------
# MaskBlock (deep-input feature modulation, from rankmixer_blocks.py)
# --------------------------------------------------------------------------

class MaskBlock(nn.Module):
    def __init__(self, input_dim: int, mask_dim: int, ratio: int = 2, std: float = 0.01):
        super().__init__()
        self.ratio = ratio
        mid = input_dim // self.ratio
        feature_emb_dim = input_dim + mask_dim
        self.mask_0 = nn.Linear(feature_emb_dim, mid)
        self.mask_1 = nn.Linear(mid, input_dim)
        _init_linear(self.mask_0, std=std)
        _init_linear(self.mask_1, std=std)

    def forward(self, input: torch.Tensor, mask_feature_inputs: torch.Tensor) -> torch.Tensor:
        feature_emb = torch.cat([input, mask_feature_inputs], dim=1)
        mask = self.mask_0(feature_emb)
        mask = torch.relu(mask)
        mask = self.mask_1(mask)
        mask = torch.sigmoid(mask) * 2.0
        return mask * input


# --------------------------------------------------------------------------
# PLE / MMoE output head (ported from model/ple.py)
# --------------------------------------------------------------------------

class Bias(nn.Module):
    def __init__(self, units: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(units))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class MLPWithBatchNorm(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bn_before: bool = False,
                 bn_after: bool = False, activation="relu",
                 activation_params: Optional[dict] = None, use_bias: bool = True,
                 std: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bn_before = nn.BatchNorm1d(input_dim, momentum=0.01, eps=1e-3) if bn_before else nn.Identity()
        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)
        _init_linear(self.linear, std=std)
        ap = dict(activation_params or {})
        if activation and str(activation).lower() == "dice":
            ap.setdefault("hidden_size", output_dim)
            ap.update({"eps": 1e-09, "momentum": 0.01, "affine": False})
        self.activation = get_activation(activation, ap)
        self.bn_after = nn.BatchNorm1d(output_dim, momentum=0.01, eps=1e-3) if bn_after else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = self.bn_before(x.reshape(-1, self.input_dim)).view(*shape)
        x = self.linear(x)
        x = self.activation(x)
        shape = x.shape
        x = self.bn_after(x.reshape(-1, self.output_dim)).view(*shape)
        return x


class MMoEGatedMLP(nn.Module):
    def __init__(self, gate_num: int, input_dim: int, output_dim: int,
                 bn_before: bool = False, bn_after: bool = False,
                 activation="relu", activation_params: Optional[dict] = None,
                 std: float = 0.01):
        super().__init__()
        self.gate_num = gate_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bn_before = nn.BatchNorm1d(input_dim, momentum=0.01, eps=1e-3) if bn_before else nn.Identity()
        self.weight = nn.Parameter(torch.empty(gate_num, input_dim, output_dim))
        nn.init.normal_(self.weight, mean=0.0, std=std)
        self.bias = nn.Parameter(torch.zeros(1, gate_num, output_dim))
        ap = dict(activation_params or {})
        if activation and str(activation).lower() == "dice":
            ap.setdefault("hidden_size", output_dim)
            ap.update({"eps": 1e-09, "momentum": 0.01, "affine": False})
        self.activation = get_activation(activation, ap)
        self.bn_after = nn.BatchNorm1d(output_dim, momentum=0.01, eps=1e-3) if bn_after else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = self.bn_before(x.reshape(-1, self.input_dim)).view(*shape)
        x = torch.einsum('bnd,ndk->bnk', x, self.weight) + self.bias
        x = self.activation(x)
        shape = x.shape
        x = self.bn_after(x.reshape(-1, self.output_dim)).view(*shape)
        return x


class MMoEMaskedGate(nn.Module):
    def __init__(self, mmoe_units: dict, gate_num: int, input_dim: int,
                 masked_dim: int, output_dim: int, ratio: int = 2, std: float = 0.01):
        super().__init__()
        num_units = len(mmoe_units['units'])
        units = mmoe_units['units']
        self.mmoe_layers = nn.ModuleList([
            MMoEGatedMLP(
                gate_num=gate_num,
                input_dim=input_dim if i == 0 else units[i - 1],
                output_dim=units[i],
                bn_before=mmoe_units['bn_before'][i],
                bn_after=mmoe_units['bn_after'][i],
                activation=mmoe_units['activation'][i],
                std=std,
            )
            for i in range(num_units)
        ])
        feature_emb_dim = units[-1] + masked_dim
        share_inputs_dim = units[-1]
        self.mask_block = MMoEMaskBlock(gate_num, feature_emb_dim, share_inputs_dim, ratio, std=std)

    def forward(self, x: torch.Tensor, masked_feature: torch.Tensor) -> torch.Tensor:
        for layer in self.mmoe_layers:
            x = layer(x)
        share_inputs = x
        feature_emb = torch.cat([x, masked_feature], dim=-1)
        feature_mask = self.mask_block(feature_emb)
        return 2.0 * feature_mask * share_inputs


class MMoEMaskBlock(nn.Module):
    """PLE MaskBlock: gate-wise mask modulation over shared expert output."""

    def __init__(self, gate_num: int, input_dim: int, output_dim: int,
                 ratio: int = 2, std: float = 0.01):
        super().__init__()
        self.weight1 = nn.Parameter(torch.empty(gate_num, input_dim, output_dim // ratio))
        nn.init.normal_(self.weight1, mean=0.0, std=std)
        self.bias1 = nn.Parameter(torch.zeros(1, gate_num, output_dim // ratio))
        self.weight2 = nn.Parameter(torch.empty(gate_num, output_dim // ratio, output_dim))
        nn.init.normal_(self.weight2, mean=0.0, std=std)
        self.bias2 = nn.Parameter(torch.zeros(1, gate_num, output_dim))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum('bnd,ndk->bnk', x, self.weight1) + self.bias1
        x = self.relu(x)
        x = torch.einsum('bnd,ndk->bnk', x, self.weight2) + self.bias2
        return self.sigmoid(x)


class PLETaskGroup(nn.Module):
    def __init__(self, group_name: str, task_group: list, mmoe_dim: int,
                 group_index: int, std: float = 0.01):
        super().__init__()
        self.group_name = group_name
        self.task_group = task_group
        self.group_index = group_index

        task_small_units = {"units": [128], "activation": ["dice"],
                            "bn_before": [True], "bn_after": [True]}
        task_units = {"units": [64, 64], "activation": ["dice", "None"],
                      "bn_before": [True, False], "bn_after": [False, False]}
        task_units['units'][-1] = len(self.task_group)

        self.small_layers = nn.ModuleList()
        for i in range(len(task_small_units['units'])):
            self.small_layers.append(MLPWithBatchNorm(
                input_dim=2 * mmoe_dim if i == 0 else task_small_units["units"][i - 1],
                output_dim=task_small_units["units"][i],
                bn_before=task_small_units["bn_before"][i],
                bn_after=task_small_units["bn_after"][i],
                activation=task_small_units["activation"][i],
                std=std,
            ))

        if self.group_name == 'interact_group':
            input_dim = mmoe_dim * 3 + 128 + 32 * 5
        else:
            input_dim = mmoe_dim * 3 + 128
        self.normal_layers = nn.ModuleList()
        for i in range(len(task_units['units'])):
            self.normal_layers.append(MLPWithBatchNorm(
                input_dim=input_dim if i == 0 else task_units["units"][i - 1],
                output_dim=task_units["units"][i],
                bn_before=task_units["bn_before"][i],
                bn_after=task_units["bn_after"][i],
                activation=task_units["activation"][i],
                std=std,
            ))

        self.task_layers = nn.ModuleDict()
        for task in task_group:
            task_layers = [Bias(1), get_activation(None)]
            self.task_layers[task] = nn.Sequential(*task_layers)

    def forward(self, x: torch.Tensor, object_emb: dict) -> dict:
        # x: (B, 256, 4, gate_num)
        task_add = x[:, :, 0, 0] + x[:, :, 0, self.group_index]
        task_dot = x[:, :, 1, 0] * x[:, :, 1, self.group_index]
        task_sub = x[:, :, 2, 0] - x[:, :, 2, self.group_index]
        task_pooling = torch.cat([x[:, :, 3, 0], x[:, :, 3, self.group_index]], dim=1)
        for layer in self.small_layers:
            task_pooling = layer(task_pooling)

        if self.group_name == 'interact_group':
            task_seq = torch.cat([
                object_emb["like"], object_emb["follow"], object_emb["share"],
                object_emb["collect"], object_emb["comment_read"],
            ], dim=-1).reshape((-1, 32 * 5))
            task_input = torch.cat([task_add, task_dot, task_sub, task_pooling, task_seq], dim=-1)
        else:
            task_input = torch.cat([task_add, task_dot, task_sub, task_pooling], dim=-1)

        for layer in self.normal_layers:
            task_input = layer(task_input)
        task_nn_pred_dict = {}
        for i, task in enumerate(self.task_group):
            pred = task_input[:, i]
            pred = self.task_layers[task](pred.unsqueeze(1)).squeeze(1)
            task_nn_pred_dict[task] = pred
        return task_nn_pred_dict


class PLE(nn.Module):
    """PLE output head: MMoE masked gate (gate_num experts) -> task groups."""

    # Production task layout (config/features/params.json task_group).
    DEFAULT_TASK_GROUPS = {
        "ctr_group": ["ctr", "ctr_top2", "impression_add4", "slide", "imp_scroll_down_top2"],
        "time_group": ["dwell", "time_percent", "wtd_short", "wtd_eff", "wtd_long",
                       "wtd_deep", "page_dwell"],
        "interact_group": ["like", "follow", "share", "collect", "dislike",
                           "comment_read", "comment_send", "highlight_point",
                           "related_searches", "poi_clk", "poi_collect"],
        "video_group": ["lpv_imp", "lpv_fin", "lpv_imp_total", "lpv_imp_2",
                        "lpv_imp_5", "lpv_imp_15", "lpv_time_6", "lpv_time_30",
                        "lpv_time_150"],
        "poi_group": ["poi_group_order", "poi_coupon", "poi_pay_bill", "poi_dwell"],
    }

    def __init__(self, masked_dim: int, mmoe_input_dim: int, gate_num: int = 6,
                 task_groups: Optional[dict] = None, std: float = 0.01):
        super().__init__()
        self.masked_dim = masked_dim
        self.gate_num = gate_num
        self.gate_tile = 4
        assert gate_num >= 2, "gate_num must include base + at least one group"
        self.task_groups = task_groups or dict(self.DEFAULT_TASK_GROUPS)

        mmoe_units = {"units": [1024, 1024], "activation": ["dice", "dice"],
                      "bn_before": [False, False], "bn_after": [True, True]}
        mmoe_output_dim = mmoe_units["units"][-1]
        self.mmoe_input_dim = mmoe_input_dim
        self.mmoe_dim = mmoe_output_dim // self.gate_tile  # 1024 // 4 = 256

        self.mmoe_masked_gate = MMoEMaskedGate(
            mmoe_units=mmoe_units, gate_num=gate_num, input_dim=mmoe_input_dim,
            masked_dim=masked_dim, output_dim=mmoe_output_dim, ratio=2, std=std,
        )

        self.ple_groups = nn.ModuleDict()
        index = 1
        for group_name, group_tasks in self.task_groups.items():
            if index >= gate_num:
                break
            self.ple_groups[group_name] = PLETaskGroup(
                group_name, group_tasks, self.mmoe_dim, index, std=std)
            index += 1
        self.num_tasks = sum(len(g.task_group) for g in self.ple_groups.values())

    def forward(self, deep_inputs: torch.Tensor, mask_feature_inputs_par: torch.Tensor,
                insert_w: Optional[torch.Tensor] = None,
                object_emb_dct: Optional[dict] = None) -> dict:
        # (B, hidden_size) -> (B, gate_num, hidden_size)
        x = deep_inputs.unsqueeze(1).expand(-1, self.gate_num, -1)
        x = self.mmoe_masked_gate(x, mask_feature_inputs_par)
        # (B, gate_num, 1024) -> (B, 256, 4, gate_num)
        x = x.permute(0, 2, 1).reshape((-1, self.mmoe_dim, self.gate_tile, self.gate_num))

        object_emb_dct = object_emb_dct or {}
        bs = x.shape[0]
        # Synthetic object embeddings for interact_group (32-dim each).
        device = x.device
        dtype = x.dtype
        if not object_emb_dct:
            object_emb_dct = {
                name: torch.zeros(bs, 32, device=device, dtype=dtype)
                for name in ["like", "follow", "share", "collect", "comment_read"]
            }

        task_nn_pred_dict: dict = {}
        for group_name, layer in self.ple_groups.items():
            group_pred = layer(x, object_emb_dct)
            task_nn_pred_dict.update(group_pred)
        # Apply the ctr position bias (matches production insert_w[0] * launch_type;
        # launch_type is approximated as 1.0 here so the bias is exercised).
        if insert_w is not None and "ctr" in task_nn_pred_dict:
            task_nn_pred_dict["ctr"] = task_nn_pred_dict["ctr"] + insert_w[0]
        if insert_w is not None and "ctr_top2" in task_nn_pred_dict:
            task_nn_pred_dict["ctr_top2"] = task_nn_pred_dict["ctr_top2"] + insert_w[1]
        return task_nn_pred_dict


# --------------------------------------------------------------------------
# Full RankMixer compute module
# --------------------------------------------------------------------------

class RankMixerArch(nn.Module):
    """RankMixer dense compute: MaskBlock -> LT -> [TokenMixer+PFFN]xN -> mean -> PLE.

    Inputs:
      embedded_sparse: [B, num_sparse_features, embedding_dim] from the embedding
        module (RecStore PS or local dynamic backend).
      mask_features: [B, masked_dim] modulation features (synthetic when absent).
    Output:
      dict[str, [B]] of per-task logits.
    """

    def __init__(self, embedding_dim: int, num_sparse_features: int,
                 segment_dims: list[int], tokens_split_dim: int = 2400,
                 rankmixer_blocks: int = 2, gate_num: int = 6,
                 masked_dim: int = 56, std: float = 0.01,
                 task_groups: Optional[dict] = None, device=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_sparse_features = num_sparse_features
        self.segment_dims = list(segment_dims)
        self.deep_input_dim = sum(self.segment_dims)
        self.tokens_split_dim = tokens_split_dim
        self.masked_dim = masked_dim

        # Flatten sparse embeddings and pad/slice to deep_input_dim.
        raw_dim = num_sparse_features * embedding_dim
        self._needs_reshape = raw_dim != self.deep_input_dim

        # MaskBlock over the concatenated deep input.
        self.mask_block = MaskBlock(input_dim=self.deep_input_dim,
                                    mask_dim=masked_dim, ratio=2, std=std)

        # Per-segment LT projection -> [B, num_tokens, tokens_split_dim].
        self._lt_layers = nn.ModuleList(
            [nn.Linear(in_dim, tokens_split_dim, bias=False) for in_dim in self.segment_dims]
        )
        for layer in self._lt_layers:
            _init_linear(layer, std=std)

        # TokenMixer + PFFN blocks.
        num_tokens = len(self.segment_dims)
        self.tokenmixer_blocks = nn.ModuleList()
        self.pffn_blocks = nn.ModuleList()
        for _ in range(rankmixer_blocks):
            self.tokenmixer_blocks.append(TokenMixer(tokens_split_dim, num_tokens))
            self.pffn_blocks.append(PFFN(
                token_num=num_tokens, token_dim=tokens_split_dim,
                hidden_size=tokens_split_dim, activation="relu",
                enable_batch_ln=True, scale_factor=3,
            ))

        # insert_w bias (matches RankMixerBlocks.insert_w).
        self.insert_w = nn.Parameter(torch.full((2,), -0.1))

        # PLE output head consumes mean-pooled token -> [B, tokens_split_dim].
        self.ple = PLE(masked_dim=masked_dim, mmoe_input_dim=tokens_split_dim,
                       gate_num=gate_num, task_groups=task_groups, std=std)
        self.gate_num = gate_num

        if device is not None:
            self.to(device)

    def to_deep_inputs(self, embedded_sparse: torch.Tensor) -> torch.Tensor:
        """Concatenate [B, F, D] -> [B, deep_input_dim] (pad/slice as needed)."""
        bs = embedded_sparse.shape[0]
        flat = embedded_sparse.reshape(bs, -1)
        if flat.shape[1] == self.deep_input_dim:
            return flat
        if flat.shape[1] > self.deep_input_dim:
            return flat[:, :self.deep_input_dim]
        pad = torch.zeros(bs, self.deep_input_dim - flat.shape[1],
                          device=flat.device, dtype=flat.dtype)
        return torch.cat([flat, pad], dim=1)

    def forward(self, embedded_sparse: torch.Tensor,
                mask_features: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> dict:
        bs = embedded_sparse.shape[0]
        device = embedded_sparse.device
        dtype = embedded_sparse.dtype
        if mask_features is None:
            mask_features = torch.zeros(bs, self.masked_dim, device=device, dtype=dtype)

        deep_inputs = self.to_deep_inputs(embedded_sparse)

        # Mask modulation.
        deep_inputs = self.mask_block(deep_inputs, mask_features)

        # Per-segment LT projection -> [B, num_tokens, tokens_split_dim].
        lengths_cumsum = [sum(self.segment_dims[:i + 1]) for i in range(len(self.segment_dims))]
        segments = [
            deep_inputs[:, lengths_cumsum[i - 1] if i > 0 else 0: lengths_cumsum[i]]
            for i in range(len(self.segment_dims))
        ]
        segments = [self._lt_layers[i](segments[i]) for i in range(len(segments))]
        orig_trans_input = torch.stack(segments, dim=1)

        for i in range(len(self.tokenmixer_blocks)):
            orig_trans_input = self.tokenmixer_blocks[i](orig_trans_input)
            orig_trans_input = self.pffn_blocks[i](orig_trans_input)

        deep_inputs = torch.mean(orig_trans_input, dim=1)
        mask_feature_inputs_par = mask_features.unsqueeze(1).expand(-1, self.gate_num, -1)
        return self.ple(deep_inputs, mask_feature_inputs_par, insert_w=self.insert_w)


# --------------------------------------------------------------------------
# Multi-task loss (ported from model/loss.py semantics)
# --------------------------------------------------------------------------

class RankMixerLoss(nn.Module):
    """Weighted multi-task loss: logloss for ctr-like tasks, mse for regression tasks.

    Task loss type / weight follow config/tasks/parameters.yml task_params.  When a
    task label is absent from the labels dict it is skipped.
    """

    # loss_type per production task_params (logloss vs mse) with representative weights.
    TASK_LOSS_CFG = {
        "ctr": ("logloss", 1.0), "ctr_top2": ("logloss", 1.0),
        "impression_add4": ("logloss", 1.0), "slide": ("logloss", 1.0),
        "imp_scroll_down_top2": ("logloss", 1.0),
        "dwell": ("mse", 0.008), "time_percent": ("mse", 1.0),
        "wtd_short": ("logloss", 0.02), "wtd_eff": ("logloss", 0.02),
        "wtd_long": ("logloss", 0.02), "wtd_deep": ("logloss", 0.02),
        "page_dwell": ("mse", 0.008),
        "like": ("logloss", 1.0), "follow": ("logloss", 1.0), "share": ("logloss", 1.0),
        "collect": ("logloss", 1.0), "dislike": ("logloss", 1.0),
        "comment_read": ("logloss", 1.0), "comment_send": ("logloss", 1.0),
        "highlight_point": ("logloss", 1.0), "related_searches": ("logloss", 1.0),
        "poi_clk": ("logloss", 1.0), "poi_collect": ("logloss", 1.0),
        "lpv_imp": ("logloss", 1.0), "lpv_fin": ("logloss", 1.0),
        "lpv_imp_total": ("mse", 1.0), "lpv_imp_2": ("logloss", 1.0),
        "lpv_imp_5": ("logloss", 1.0), "lpv_imp_15": ("logloss", 1.0),
        "lpv_time_6": ("mse", 1.0), "lpv_time_30": ("mse", 1.0),
        "lpv_time_150": ("mse", 1.0),
        "poi_group_order": ("logloss", 1.0), "poi_coupon": ("logloss", 1.0),
        "poi_pay_bill": ("logloss", 1.0), "poi_dwell": ("mse", 0.008),
    }

    def __init__(self, task_names: list[str]):
        super().__init__()
        self.task_names = task_names
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: dict, labels: dict) -> torch.Tensor:
        total = None
        for task in self.task_names:
            if task not in logits or labels.get(task) is None:
                continue
            pred = logits[task]
            tgt = labels[task].to(pred.dtype).view_as(pred)
            loss_type, weight = self.TASK_LOSS_CFG.get(task, ("logloss", 1.0))
            if loss_type == "mse":
                loss = nn.functional.mse_loss(pred, tgt)
            else:
                loss = self.bce(pred, tgt)
            loss = loss * weight
            total = loss if total is None else total + loss
        if total is None:
            # Fallback: use the first available task.
            task = self.task_names[0]
            pred = logits[task]
            tgt = labels[task].to(pred.dtype).view_as(pred)
            total = self.bce(pred, tgt)
        return total


def default_segment_dims(num_sparse_features: int, embedding_dim: int,
                         num_segments: int = 5) -> list[int]:
    """Partition num_sparse_features*embedding_dim into num_segments segments."""
    total = num_sparse_features * embedding_dim
    base = total // num_segments
    dims = [base] * num_segments
    dims[-1] += total - base * num_segments
    return dims


def build_rankmixer_arch(embedding_dim: int, num_sparse_features: int,
                         segment_dims: Optional[list[int]] = None,
                         tokens_split_dim: int = 2400, rankmixer_blocks: int = 2,
                         gate_num: int = 6, masked_dim: int = 56,
                         device=None) -> RankMixerArch:
    if segment_dims is None:
        segment_dims = default_segment_dims(num_sparse_features, embedding_dim)
    return RankMixerArch(
        embedding_dim=embedding_dim,
        num_sparse_features=num_sparse_features,
        segment_dims=segment_dims,
        tokens_split_dim=tokens_split_dim,
        rankmixer_blocks=rankmixer_blocks,
        gate_num=gate_num,
        masked_dim=masked_dim,
        device=device,
    )
