"""
Step 02: 因果自注意力（单头）

Attention 是 token 之间交换信息的唯一途径。每个 token 产生三个向量：
    q (query): "我在找什么信息？"
    k (key):   "我这里有什么信息？"（用来被别人匹配）
    v (value): "如果你选中了我，我给你的内容"

对第 i 个 token：
    1. 打分:   score[i, j] = q_i · k_j / sqrt(d_head)      i 和每个 j 有多"匹配"
    2. 因果:   j > i 的位置分数设为 -inf                    不许偷看未来
    3. 归一化: weight[i, :] = softmax(score[i, :])         变成加起来为 1 的权重
    4. 汇总:   out_i = Σ_j weight[i, j] * v_j              按权重混合各位置的 value

写成矩阵:  out = softmax(mask(Q @ K^T / sqrt(d_head))) @ V

注意：本文件里的函数要支持任意数量的"前置维度"：
    q, k, v 可以是 [B, T, d]，也可以是 [B, H, T, d]（Step 03 的多头会用到）。
    所以请用倒数的维度（-1, -2）来写，不要写死 dim=1、dim=2。

测试: uv run pytest tests/test_02_attention.py -v
"""

import math

import torch
from torch import Tensor, nn


def causal_attention_weights(q: Tensor, k: Tensor) -> Tensor:
    """q, k: [..., T, d_head]  ->  weights: [..., T, T]

    weights[..., i, j] 是第 i 个 token 对第 j 个 token 的注意力权重：
    每一行加起来等于 1，且 j > i 的位置为 0。
    """
    # TODO 1: 打分 -> 缩放 -> 因果 mask -> softmax
    #
    #   会用到的 API:
    #     k.transpose(-2, -1)          交换最后两个维度: [..., T, d] -> [..., d, T]
    #     a @ b                        矩阵乘法，前置维度会自动对齐（批量矩阵乘）
    #     math.sqrt(x)
    #     torch.ones(T, T, dtype=torch.bool, device=q.device)
    #     torch.triu(m, diagonal=1)    只保留对角线"上方"的部分（不含对角线），其余置 0/False
    #                                  —— 这正好是 "j > i" 的位置
    #     t.masked_fill(mask, value)   把 mask 为 True 的位置填成 value，返回新 tensor
    #     float("-inf")                softmax 之后 e^(-inf) = 0，这就是"屏蔽"的原理
    #     torch.softmax(t, dim=-1)
    t, d_head = q.shape[-2], q.shape[-1]
    score = q @ k.transpose(-1, -2) / math.sqrt(d_head)
    mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), diagonal=1)
    score_masked = score.masked_fill(mask, float("-inf"))
    weight = torch.softmax(score_masked, dim=-1)

    return weight


def causal_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """q, k, v: [..., T, d_head]  ->  out: [..., T, d_head]"""
    # TODO 2: 用上面的权重对 v 加权求和（一行就够）
    weight = causal_attention_weights(q, k)
    return weight @ v


class SingleHeadAttention(nn.Module):
    """完整的单头注意力层: x [B, T, d_model] -> [B, T, d_model]"""

    def __init__(self, d_model: int, d_head: int) -> None:
        super().__init__()
        # TODO 3: 四个线性层，属性名必须是 q_proj / k_proj / v_proj / out_proj
        #           q_proj, k_proj, v_proj: d_model -> d_head   （从 x 算出 q、k、v）
        #           out_proj:               d_head  -> d_model  （把结果映射回残差流的维度）
        #
        #   会用到的 API:
        #     nn.Linear(in_features, out_features)
        #         就是 y = x @ W.T + b，W 形状 [out, in]。它作用在最后一维上，
        #         所以 [B, T, in] 进去，[B, T, out] 出来。
        self.q_proj = nn.Linear(d_model, d_head)
        self.k_proj = nn.Linear(d_model, d_head)
        self.v_proj = nn.Linear(d_model, d_head)

        self.out_proj = nn.Linear(d_head, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # TODO 4: x -> q, k, v -> causal_attention -> out_proj
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        attention = causal_attention(q, k, v)

        out = self.out_proj(attention)

        return out


# ===============================================================
# Step 03: 多头注意力
#
# 思路：不是把 x 切开，而是每个头都读取"完整的 x"，各自用自己的 W_q、W_k、W_v
# 投影出一组小的 q、k、v（d_head = d_model / n_heads 维），各自算一个注意力矩阵。
#
# 实现技巧：H 个头的 [d_head, d_model] 投影矩阵上下叠在一起，正好是一个
# [d_model, d_model] 的矩阵。所以用一个大的 nn.Linear 一次算出所有头的 q，
# 再把结果的最后一维按顺序切成 H 段，每段 d_head 维，就是每个头的 q。
#
#     x [B, T, d_model] --q_proj--> [B, T, d_model] --split--> [B, H, T, d_head]
#
# 把 H 放到 T 前面之后，Step 02 的 causal_attention 可以原封不动地用：
# 它把 [B, H] 都当作"前置维度"，等于同时算了 B*H 个独立的注意力。
#
# 测试: uv run pytest tests/test_03_multihead.py -v
# ===============================================================
class MultiHeadAttention(nn.Module):
    """x [B, T, d_model] -> [B, T, d_model]"""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        # TODO 1: 检查 d_model 能被 n_heads 整除（用 assert），
        #         保存 self.n_heads 和 self.d_head，
        #         创建四个线性层 q_proj / k_proj / v_proj / out_proj，都是 d_model -> d_model

    def split_heads(self, x: Tensor) -> Tensor:
        """[B, T, d_model] -> [B, H, T, d_head]"""
        # TODO 2: 把最后一维按顺序切成 H 段，再把 H 这个维度挪到 T 前面
        #
        #   会用到的 API:
        #     x.view(B, T, H, d_head)      不复制数据，只是换一种方式"看"同一块内存。
        #                                  最后一维 d_model 被拆成 [H, d_head]：
        #                                  前 d_head 个数归第 0 个头，接下来 d_head 个归第 1 个头……
        #     x.transpose(1, 2)            交换第 1、2 维: [B, T, H, d_head] -> [B, H, T, d_head]
        raise NotImplementedError

    def merge_heads(self, x: Tensor) -> Tensor:
        """[B, H, T, d_head] -> [B, T, d_model]，split_heads 的逆操作"""
        # TODO 3: 先把 H 挪回去，再把最后两维合并
        #
        #   会用到的 API:
        #     x.transpose(1, 2)
        #     x.reshape(B, T, d_model)     和 view 类似，但在内存不连续时会自动复制一份。
        #                                  transpose 之后内存就不连续了，这里直接用 view 会报错
        #                                  （也可以写 x.contiguous().view(...)）
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        # TODO 4: 投影 -> split_heads -> causal_attention -> merge_heads -> out_proj
        raise NotImplementedError

    def attention_pattern(self, x: Tensor) -> Tensor:
        """返回每个头的注意力权重 [B, H, T, T]。以后做可解释性实验（画 attention 图）用。"""
        # TODO 5: 和 forward 的前半段一样，但调用的是 causal_attention_weights
        raise NotImplementedError
