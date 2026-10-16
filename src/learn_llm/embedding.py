"""
Step 01: Embedding —— 把 token 编号变成向量

模型的第一步：
    token_ids [batch, seq]（整数编号）  ->  x [batch, seq, d_model]（向量）

GPT-2 的做法是两张"表"相加：
    - wte (word token embedding):  [vocab_size, d_model]
      第 i 行就是编号为 i 的 token 的向量。"查表"就是按编号取出对应的行。
    - wpe (word position embedding): [max_seq_len, d_model]
      第 t 行是"第 t 个位置"的向量。attention 本身不知道 token 的顺序，
      必须把位置信息加进去，否则 "狗咬人" 和 "人咬狗" 对模型来说是一样的。

    x[b, t] = wte[token_ids[b, t]] + wpe[t]

规则：不许用 nn.Embedding，自己用 nn.Parameter + 下标索引实现"查表"。

测试: uv run pytest tests/test_01_embedding.py
"""

import torch
from torch import Tensor, nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        # TODO 1: 创建两张可学习的表，属性名必须是 self.wte 和 self.wpe
        #         （测试和之后加载 GPT-2 官方权重都靠这两个名字）
        #           self.wte: [vocab_size, d_model]
        #           self.wpe: [max_seq_len, d_model]
        #
        #   会用到的 API:
        #     torch.randn(a, b)            标准正态分布的随机 tensor，形状 [a, b]
        #     t * 0.02                     GPT-2 初始化时用的标准差是 0.02
        #     nn.Parameter(t)              把 tensor 包装成"可学习参数"，
        #                                  这样它才会出现在 model.parameters() 里、被优化器更新
        wte = torch.randn(vocab_size, d_model)
        wte *= 0.02
        wpe = torch.randn(max_seq_len, d_model)
        wpe *= 0.02

        self.wte = nn.Parameter(wte)
        self.wpe = nn.Parameter(wpe)

    def forward(self, token_ids: Tensor) -> Tensor:
        """token_ids: [batch, seq] 整数  ->  [batch, seq, d_model]"""
        # TODO 2: 查表 + 加上位置向量
        #
        #   会用到的 API:
        #     table[idx]                   用整数 tensor 当下标（"高级索引"）：
        #                                  idx 形状是 [B, T]，table 是 [V, d]，结果是 [B, T, d]
        #     t.shape                      比如 B, T = token_ids.shape
        #     torch.arange(n, device=d)    [0, 1, ..., n-1]；device 要和输入一致，
        #                                  可以写 device=token_ids.device
        #     广播（broadcasting）          [B, T, d] + [T, d] 会自动把后者复制到每个 batch 上
        wte_result = self.wte[token_ids]
        wpe_result = self.wpe[: token_ids.shape[1]]

        return wte_result + wpe_result
