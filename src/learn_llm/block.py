"""
Step 04: Transformer Block —— 把 attention 和 MLP 串起来

一个 GPT-2 的 block 由两个"子层"组成，每个子层都是"读残差流 -> 算点东西 -> 加回残差流"：

    x = x + attn(ln_1(x))      # 子层 1: token 之间交换信息（Step 02/03）
    x = x + mlp(ln_2(x))       # 子层 2: 每个 token 各自"加工"，token 之间不交流

整个过程中形状始终是 [B, T, d_model] —— 这样 block 才能一层层叠起来（GPT-2 small 叠了 12 层）。

    x [B, T, d_model]
    ├─ ln_1 ─ attn ─┐            attn:  [B, T, d_model] -> [B, T, d_model]
    x ◄──── + ──────┘
    ├─ ln_2 ─ mlp ──┐            mlp:   [B, T, d_model] -> [B, T, 4*d_model] -> [B, T, d_model]
    x ◄──── + ──────┘
    out [B, T, d_model]

三个概念：
    残差流 (residual stream):  x 本身这条"主干道"。子层只往里"加"东西，从不覆盖它，
                               所以就算某个子层输出 0，信息也能原样流到下一层。
    LayerNorm:                 进子层前，把每个 token 的向量标准化（均值 0、方差 1），
                               让子层看到的输入尺度稳定。直接用 nn.LayerNorm。
    pre-LN:                    LayerNorm 放在子层"入口"，残差加的是没归一化过的 x 本身。

测试: uv run pytest tests/test_04_block.py -v
"""

from torch import Tensor, nn

from learn_llm.attention import MultiHeadAttention


class MLP(nn.Module):
    """x [B, T, d_model] -> [B, T, d_model]

    先升维到 4 倍，过激活函数，再降回来。
    nn.Linear 只作用在最后一维上，所以每个 token 是被"单独"处理的 ——
    MLP 里 token 之间完全没有信息交流（交流只发生在 attention 里）。
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        # TODO 1: 三个子模块，属性名必须是 fc / gelu / proj
        #           fc:   d_model   -> 4*d_model   （升维）
        #           gelu: 激活函数，一个"圆角版"的 ReLU
        #           proj: 4*d_model -> d_model     （降维，写回残差流）
        #
        #   会用到的 API:
        #     nn.Linear(in_features, out_features)
        #     nn.GELU(approximate="tanh")    GPT-2 用的是 tanh 近似版的 GELU
        self.fc = nn.Linear(d_model, 4*d_model)
        self.gelu = nn.GELU("tanh")
        self.proj = nn.Linear(4*d_model,  d_model)

    def forward(self, x: Tensor) -> Tensor:
        # TODO 2: fc -> gelu -> proj
        return self.proj(self.gelu(self.fc(x)))


class Block(nn.Module):
    """一个完整的 Transformer block: x [B, T, d_model] -> [B, T, d_model]"""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        # TODO 3: 四个子模块，属性名必须是 ln_1 / attn / ln_2 / mlp（和 GPT-2 官方权重的名字对应）
        #           ln_1: 子层 1（attn）入口的 LayerNorm
        #           attn: Step 03 写的 MultiHeadAttention
        #           ln_2: 子层 2（mlp）入口的 LayerNorm
        #           mlp:  上面的 MLP
        #
        #   会用到的 API:
        #     nn.LayerNorm(d_model)          对最后一维做标准化，自带可学习的 weight / bias
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)


    def forward(self, x: Tensor) -> Tensor:
        # TODO 4: 两个子层，每个都是 "x = x + 子层(ln(x))"
        #
        #   注意：是 x + 子层(ln(x))，不是 ln(x + 子层(x))（后者是原版 Transformer 的 post-LN）。
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
