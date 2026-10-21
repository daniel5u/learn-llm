"""
Step 06: 生成文本 —— 把"下一个 token 的打分"变成一段话

模型本身只会做一件事：给定前面的 token，对"下一个 token"打分。生成就是反复做这件事：

    ids [B, T]
      │ model                      （Step 05）
      ▼
    logits [B, T, V]
      │ 只取最后一个位置 [:, -1, :]  它看过了整句话，预测的是"句子后面那个"
      ▼
    logits [B, V]
      │ / temperature              调整"自信程度"
      │ top-k 过滤                  只在最靠谱的 k 个候选里选
      │ softmax -> 抽样             打分变成概率，按概率随机抽一个
      ▼
    next_id [B, 1]
      │ 拼到 ids 后面
      ▼
    ids [B, T+1]  ──> 回到开头，再来一轮

几个旋钮：
    temperature:  logits / temperature 再 softmax。
                  < 1 让分布更"尖"（更保守、更重复），> 1 让分布更"平"（更随机、更容易胡言乱语）。
                  = 0 时约定为贪心解码（greedy）：永远选打分最高的那个，结果是确定的。
    top_k:        只保留打分最高的 k 个 token，其余的概率直接归零。
                  防止从几万个"长尾"烂选项里偶然抽中一个，把整句话带歪。

注意：这样写每生成一个 token 都要把整个序列重新算一遍，很浪费 —— 前面 token 的 k、v 其实没变。
真正的推理会用 KV cache 把它们存起来，这是以后的优化。

测试: uv run pytest tests/test_06_generate.py -v
试玩: uv run python -m learn_llm.generate "The meaning of life is"
"""

import sys

import torch
from torch import Tensor

from learn_llm.gpt import GPT


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """logits [..., V] -> [..., V]：每一行只保留最大的 k 个值，其余改成 -inf。

    -inf 经过 softmax 就是概率 0，这些 token 永远不会被抽到。
    """
    # TODO 1: 找出每一行第 k 大的值，把比它小的位置都填成 -inf
    #
    #   会用到的 API:
    #     torch.topk(t, k, dim=-1)       返回 (values, indices)，values 从大到小排好
    #     values[..., -1:]               第 k 大的值；用切片 -1: 而不是 -1，保留最后一维方便广播
    #     t.masked_fill(mask, value)
    #     float("-inf")
    values, _= torch.topk(logits, k, -1)
    k_value = values[..., -1:]
    logits = logits.masked_fill(logits < k_value, float('-inf'))
    return logits


def sample_next_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """logits [B, V] -> next_id [B, 1]"""
    # TODO 2: 按顺序处理
    #           temperature == 0  ->  直接取 argmax（贪心）
    #           否则              ->  除以 temperature -> （若给了 top_k）top_k_filter -> softmax -> 抽样
    #
    #   会用到的 API:
    #     t.argmax(dim=-1, keepdim=True)                   [B, V] -> [B, 1]
    #     torch.softmax(t, dim=-1)
    #     torch.multinomial(probs, 1, generator=generator) 每一行按概率抽 1 个下标: [B, V] -> [B, 1]
    #                                                      generator 用来固定随机种子，让结果可复现
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    else:
        logits = logits / temperature

        if top_k:
            logits = top_k_filter(logits, top_k)

        logits = torch.softmax(logits, -1)
    return torch.multinomial(logits, num_samples=1, generator=generator)



@torch.no_grad()  # 生成不需要算梯度：省内存、更快
def generate(
    model: GPT,
    token_ids: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """token_ids [B, T] -> [B, T + max_new_tokens]（前 T 个就是原来的 prompt）"""
    # TODO 3: 循环 max_new_tokens 次:
    #           1. 只把最后 max_seq_len 个 token 喂给模型（位置表 wpe 只有这么长，再长就查不到了）
    #           2. model(...) 得到 logits，只取最后一个位置
    #           3. sample_next_token 抽出下一个 token
    #           4. 拼到 token_ids 后面
    #
    #   会用到的 API:
    #     model.cfg.max_seq_len
    #     ids[:, -n:]                    取最后 n 个（不足 n 个时就是全部）
    #     logits[:, -1, :]               [B, T, V] -> [B, V]
    #     torch.cat([a, b], dim=1)       [B, T] 和 [B, 1] 拼成 [B, T+1]
    for _ in range(max_new_tokens):
        logits = model(token_ids[:, -model.cfg.max_seq_len:])[:, -1, :]
        next_token = sample_next_token(logits, temperature, top_k, generator)
        token_ids = torch.cat([token_ids, next_token], dim=1)
    return token_ids



def main() -> None:
    from learn_llm.pretrained import load_gpt2

    prompt = sys.argv[1] if len(sys.argv) > 1 else "The meaning of life is"
    model, tok = load_gpt2()
    ids = torch.tensor([tok.encode(prompt)])
    for temperature, top_k in [(0.0, None), (0.8, 50), (1.5, None)]:
        out = generate(model, ids, max_new_tokens=30, temperature=temperature, top_k=top_k)
        print(f"--- temperature={temperature}, top_k={top_k}")
        print(tok.decode(out[0]))


if __name__ == "__main__":
    main()
