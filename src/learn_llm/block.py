"""
Step 04: Transformer Block —— LayerNorm + MLP + 残差连接

一个 GPT-2 的 block 由两个"子层"组成，每个子层都是"读残差流 -> 算点东西 -> 加回残差流"：

    x = x + attn(ln_1(x))      # 子层 1: token 之间交换信息（Step 02/03）
    x = x + mlp(ln_2(x))       # 子层 2: 每个 token 各自"思考"，token 之间不交流

这叫 pre-LN（先归一化再进子层）。x 本身这条"主干道"叫残差流（residual stream）：
每个子层只是往里面"加"一点东西，从不覆盖它。所以就算某个子层什么都没学到（输出 0），
信息也能原样流到下一层 —— 这是深层网络能训练起来的关键。

本步骤的三个零件：
    LayerNorm:  把每个 token 的向量标准化成均值 0、方差 1，再乘 γ（weight）加 β（bias）
    gelu:       GPT-2 用的激活函数（一个"平滑版"的 ReLU）
    MLP:        d_model -> 4*d_model -> gelu -> d_model

规则：不许用 nn.LayerNorm / F.layer_norm / F.gelu / nn.GELU，自己实现。

测试: uv run pytest tests/test_04_block.py -v
"""

import math

import torch
from torch import Tensor, nn

from learn_llm.attention import MultiHeadAttention


class LayerNorm(nn.Module):
    """对最后一维做标准化: x [..., d_model] -> [..., d_model]

        y = (x - mean) / sqrt(var + eps) * weight + bias

    mean 和 var 是在"每个 token 自己的 d_model 个数"上算的，
    不同 token、不同 batch 之间互不影响。
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        # TODO 1: 保存 self.eps，创建两个可学习参数（属性名必须是 weight 和 bias）
        #           self.weight: [d_model]，初始化为全 1   （γ，缩放）
        #           self.bias:   [d_model]，初始化为全 0   （β，平移）
        #         这样刚初始化时 LayerNorm 就是"纯标准化"，不额外缩放平移。
        #
        #   会用到的 API:
        #     torch.ones(n) / torch.zeros(n)
        #     nn.Parameter(t)

    def forward(self, x: Tensor) -> Tensor:
        # TODO 2: 按公式实现
        #
        #   会用到的 API:
        #     x.mean(dim=-1, keepdim=True)                  [..., d] -> [..., 1]
        #                                                   keepdim 保留这一维，方便和 x 广播相减
        #     x.var(dim=-1, keepdim=True, unbiased=False)   方差。注意 unbiased=False：
        #                                                   除以 n 而不是 n-1，LayerNorm 用的是前者
        #     torch.sqrt(t)
        #     广播                                          [..., d] * [d] 会自动对齐最后一维
        raise NotImplementedError


def gelu(x: Tensor) -> Tensor:
    """GPT-2 用的 GELU（tanh 近似版），逐元素计算:

        gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x³) ))

    直觉：x 很大时 ≈ x，x 很小（很负）时 ≈ 0，中间平滑过渡 —— 像一个"圆角"的 ReLU。
    """
    # TODO 3: 照公式写（一行就够）
    #
    #   会用到的 API:
    #     torch.tanh(t)
    #     math.sqrt(2 / math.pi)
    #     x ** 3   或   x * x * x
    raise NotImplementedError


class MLP(nn.Module):
    """x [B, T, d_model] -> [B, T, d_model]

    先升维到 4 倍，过激活函数，再降回来。
    nn.Linear 只作用在最后一维上，所以每个 token 是被"单独"处理的 ——
    MLP 里 token 之间完全没有信息交流（交流只发生在 attention 里）。
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        # TODO 4: 两个线性层，属性名必须是 fc 和 proj
        #           fc:   d_model   -> 4*d_model   （升维）
        #           proj: 4*d_model -> d_model     （降维，写回残差流）

    def forward(self, x: Tensor) -> Tensor:
        # TODO 5: fc -> gelu -> proj
        raise NotImplementedError


class Block(nn.Module):
    """一个完整的 Transformer block: x [B, T, d_model] -> [B, T, d_model]"""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        # TODO 6: 四个子模块，属性名必须是 ln_1 / attn / ln_2 / mlp（和 GPT-2 官方权重的名字对应）
        #           ln_1: LayerNorm(d_model)
        #           attn: MultiHeadAttention(d_model, n_heads)   （Step 03 写的）
        #           ln_2: LayerNorm(d_model)
        #           mlp:  MLP(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # TODO 7: pre-LN + 残差，两行:
        #           x = x + attn(ln_1(x))
        #           x = x + mlp(ln_2(x))
        #
        #   注意：是 x + 子层(ln(x))，不是 ln(x + 子层(x))（后者是原版 Transformer 的 post-LN）。
        #   残差加的是"没归一化过"的 x 本身。
        raise NotImplementedError
