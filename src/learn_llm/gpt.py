"""
Step 05: 完整的 GPT 模型 —— 从 token 编号到"下一个 token 的打分"

把前面所有零件串起来：

    token_ids [B, T]                      整数编号
      │ embed      (Step 01)
      ▼
    x [B, T, d_model]                     残差流从这里开始
      │ blocks[0], blocks[1], ..., blocks[n_layers-1]   (Step 04)，形状一直不变
      ▼
    x [B, T, d_model]
      │ ln_f       最后一个 LayerNorm（pre-LN 的每个子层入口都有 LN，但残差流本身
      │            从没被归一化过，所以输出前要再归一化一次）
      ▼
    x [B, T, d_model]
      │ @ wte.T    unembedding：和词表里每个 token 的向量做点积
      ▼
    logits [B, T, vocab_size]             logits[b, t, v] = "第 t 个位置之后，下一个 token 是 v" 的打分

关键点：
    1. 每个位置都会输出一个预测。logits[:, t] 只看过 token 0..t（因果 attention），
       它预测的是第 t+1 个 token。训练时 T 个位置同时算 loss，生成时只用最后一个位置。
    2. 权重共享 (weight tying)：GPT-2 没有单独的输出层，直接复用 embedding 的 wte。
       输入时 wte 把"编号 -> 向量"，输出时 wte.T 把"向量 -> 每个编号的打分"。
       这样省了 vocab_size * d_model ≈ 3800 万个参数（GPT-2 small 总共才 1.24 亿）。

测试: uv run pytest tests/test_05_gpt.py -v
"""

from dataclasses import dataclass

from torch import Tensor, nn

from learn_llm.block import Block
from learn_llm.embedding import Embedding


@dataclass
class GPTConfig:
    """默认值就是 GPT-2 small（1.24 亿参数）。"""

    vocab_size: int = 50257
    max_seq_len: int = 1024
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12


class GPT(nn.Module):
    """token_ids [B, T] -> logits [B, T, vocab_size]"""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # TODO 1: 三个子模块，属性名必须是 embed / blocks / ln_f
        #           embed:  Step 01 的 Embedding
        #           blocks: n_layers 个 Step 04 的 Block
        #           ln_f:   最后的 LayerNorm
        #         不需要单独的输出层 —— 见上面的"权重共享"。
        #
        #   会用到的 API:
        #     nn.ModuleList([...])           装子模块的 list。必须用它而不是普通 list，
        #                                    否则里面 Block 的参数不会出现在 model.parameters() 里
        #     [Block(...) for _ in range(n)] 列表推导式，每次循环都新建一个 Block（各自有独立参数）
        #     nn.LayerNorm(d_model)
        self.embed = Embedding(self.cfg.vocab_size, self.cfg.max_seq_len, self.cfg.d_model)
        self.blocks = nn.ModuleList([Block(self.cfg.d_model, self.cfg.n_heads) for _ in range(self.cfg.n_layers)])
        self.ln_f = nn.LayerNorm(self.cfg.d_model)
        

    def forward(self, token_ids: Tensor) -> Tensor:
        # TODO 2: embed -> 逐个过 blocks -> ln_f -> 乘 wte 的转置得到 logits
        #
        #   会用到的 API:
        #     for block in self.blocks:      ModuleList 可以直接 for 循环
        #     self.embed.wte                 [vocab_size, d_model]
        #     x @ w.T                        [B, T, d] @ [d, V] -> [B, T, V]
        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = x @ self.embed.wte.T
        return x
