#
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaAttention
from contextlib import contextmanager
from dataclasses import dataclass

class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merge_weights = merge_weights


class LoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        #this is a list due to layer by layer control 
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out

        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)

        should = self.merged if mode else not self.merged
        if self.merge_weights and should:
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0),
                    self.lora_B.data.unsqueeze(-1),
                    groups=sum(self.enable_lora)
                ).squeeze(0)
                sign = -1 if mode else 1
                self.weight.data += sign * self.zero_pad(T(delta_w * self.scaling))
            self.merged = not mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(
                    after_A.transpose(-2, -1),
                    self.lora_B.unsqueeze(-1),
                    groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result

def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    if bias == "none":
        return
    elif bias == "all":
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
    elif bias == "lora_only":
        for module in model.modules():
            if isinstance(module, LoRALayer) and hasattr(module, "bias") and module.bias is not None:
                module.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = "none") -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == "none":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k}
    elif bias == "all":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        for k in my_state_dict:
            if "lora_" in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


class CausalSelfAttention(LlamaAttention):
    lora_config = None
    def __init__(self, config):
        super().__init__(config)
        self.q_proj = LoRALinear(
            config.hidden_size,
            config.num_heads * config.head_dim,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_dropout,
            enable_lora=[True],
            fan_in_fan_out=False,
            merge_weights=True,
            bias=False,
        )
        self.k_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.v_proj = LoRALinear(
            config.hidden_size,
            config.num_heads * config.head_dim,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            enable_lora=[True],
            fan_in_fan_out=False,
            merge_weights=True,
            bias=False,
        )
@dataclass
class LoRAConfig:
    r: float = 0.0
    alpha: float = 1.0
    dropout: float = 0.0

@contextmanager
def lora(r: int, alpha: float, dropout: float, enabled: bool = True):
    if not enabled:
        yield
        return

    CausalSelfAttention.lora_config = LoRAConfig(r=r, alpha=alpha, dropout=dropout)
    yield
    CausalSelfAttention.lora_config = None