# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Callable, List, Any, Tuple, Dict

class Adapter(nn.Module):
    def __init__(
        self,
        skip_connect=False,
    ) -> None:
        super().__init__()
        self.skip_connect=skip_connect
     
        # self.down1 = nn.Linear(768, 256) 
        # # self.non_linear_func = nn.ReLU()
        # self.down2 = nn.Linear(256, 128)
        # self.down3 = nn.Linear(384, 256)

        # self.up1 = nn.Linear(128, 256)
        # self.up2 = nn.Linear(256, 768)

        self.down1 = nn.Linear(768, 64) 
        # self.non_linear_func = nn.ReLU()
        self.down2 = nn.Linear(64, 32)
        self.down3 = nn.Linear(96, 64)

        self.up1 = nn.Linear(32, 64)
        self.up2 = nn.Linear(64, 768)

        with torch.no_grad():
                nn.init.kaiming_uniform_(self.down1.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.down2.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.down3.weight, a=math.sqrt(5))

                nn.init.zeros_(self.up1.weight)
                nn.init.zeros_(self.up2.weight)

             
                nn.init.zeros_(self.down1.bias)
                nn.init.zeros_(self.down2.bias)
                nn.init.zeros_(self.down3.bias)
                nn.init.zeros_(self.up1.bias)
                nn.init.zeros_(self.up2.bias)


    def forward(self, x: Tensor) -> List[Tensor]:
        x0 = self.down1(x)
        x0 = F.relu(x0, inplace=True)
        
        xs_1 = self.down2(x0)
        xs_2 = self.up1(xs_1)

        outputs = self.down3(torch.cat([xs_1,xs_2],dim=2))

        outputs += x0

        outputs = self.up2(outputs)
        if self.skip_connect:
            outputs+=x
        return outputs

# class Adapter(nn.Module):
#     def __init__(self,
#                  config=None,
#                  d_model=None,
#                  bottleneck=None,
#                  dropout=0.0,
#                  init_option="bert",
#                  adapter_scalar="1.0",
#                  adapter_layernorm_option="in"):
#         super().__init__()
#         self.n_embd = config.d_model if d_model is None else d_model
#         self.down_size = config.attn_bn if bottleneck is None else bottleneck

#         #_before
#         self.adapter_layernorm_option = adapter_layernorm_option

#         self.adapter_layer_norm_before = None
#         if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
#             self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

#         if adapter_scalar == "learnable_scalar":
#             self.scale = nn.Parameter(torch.ones(1))
#         else:
#             self.scale = float(adapter_scalar)

#         self.down_proj = nn.Linear(self.n_embd, self.down_size)
#         self.non_linear_func = nn.ReLU()
#         self.up_proj = nn.Linear(self.down_size, self.n_embd)

#         self.dropout = dropout
#         if init_option == "bert":
#             raise NotImplementedError
#         elif init_option == "lora":
#             with torch.no_grad():
#                 nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
#                 nn.init.zeros_(self.up_proj.weight)
#                 nn.init.zeros_(self.down_proj.bias)
#                 nn.init.zeros_(self.up_proj.bias)

#     def forward(self, x, add_residual=True, residual=None):
#         residual = x if residual is None else residual
#         if self.adapter_layernorm_option == 'in':
#             x = self.adapter_layer_norm_before(x)

#         down = self.down_proj(x)
#         down = self.non_linear_func(down)
#         down = nn.functional.dropout(down, p=self.dropout, training=self.training)
#         up = self.up_proj(down)

#         up = up * self.scale

#         if self.adapter_layernorm_option == 'out':
#             up = self.adapter_layer_norm_before(up)

#         if add_residual:
#             output = up + residual
#         else:
#             output = up

#         return output