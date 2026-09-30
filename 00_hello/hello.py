"""
Hello World: 偷看 LLM 的"脑子" —— 它下一个词在想什么？

运行: uv run 00_hello/hello.py
"""

import torch

# 从具体子模块导入，而不是 `from transformers import ...`：
# 后者会让编辑器的类型检查器解析整个 transformers（约 2500 个文件），非常慢
from transformers.models.qwen2 import Qwen2Tokenizer  # Qwen3 沿用 Qwen2 的 tokenizer
from transformers.models.qwen3 import Qwen3ForCausalLM

MODEL = "Qwen/Qwen3-0.6B"
# from_pretrained 的返回类型没有标注，手动标上，给编辑器补全用
tok: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(MODEL)
model = Qwen3ForCausalLM.from_pretrained(MODEL)
model.eval()

TEMPERATURE = 0.001

prompt = "Leo Messi"
print(f"Our prompt is: {prompt}")


# ---------------------------------------------------------------
# 任务 1: 看看模型"脑子里"的下一个词
# ---------------------------------------------------------------
# TODO 1a: 用 tok 把 prompt 变成 token id，打印出来。
#          顺便把每个 id 解码回文字，看看 prompt 被切成了哪几块。
encoded_prompt: list[int] = tok.encode(text=prompt)
print(f"The prompt will be encoded into tokens with ids: {encoded_prompt}")
prompt_tokens = tok.convert_ids_to_tokens(encoded_prompt)
print(f"And the ids represents: {prompt_tokens}")

# TODO 1b: 把 token id 喂给 model，拿到 logits。
#          先打印 logits.shape —— 三个维度分别代表什么？
#
#   会用到的 API:
#     torch.tensor(x)              list -> tensor
#     t.unsqueeze(0)               在最前面加一个维度（模型要求输入带 batch 维度: [batch, seq]）
#     with torch.no_grad():        推理时不需要算梯度，包在这个块里更快、更省内存
#     model(input_ids).logits      模型的输出，是一个 tensor
token_ids = torch.tensor(encoded_prompt)
token_ids = token_ids.unsqueeze(0)
with torch.no_grad():
    logits = model(token_ids).logits
    print(f"The prompt is turned into shape:{logits.shape}")

# TODO 1c: 取出"最后一个位置"的 logits，用 softmax 变成概率，
#          打印概率最高的 10 个候选词和它们的概率。
#
#   会用到的 API:
#     t[0, -1]                     tensor 索引，和 numpy 一样（这里取 batch 0、最后一个位置）
#     torch.softmax(t, dim=-1)     在最后一个维度上做 softmax
#     torch.topk(t, k=10)          返回 (values, indices) 两个 tensor
#     tok.decode(id)               id -> 文字（id 可以是 int，也可以是只有一个数的 tensor）
#     t.item()                     单元素 tensor -> Python 数字
last_position = logits[0, -1] / TEMPERATURE
soft_vec = torch.softmax(last_position, dim=-1)
(values, indicies) = torch.topk(soft_vec, k=10)
for value, index in zip(values, indicies):
    print(
        f"The probability of `{tok.decode(index):<10}` with temperature = {TEMPERATURE} is:\t{value}"
    )


# ---------------------------------------------------------------
# 任务 2: 自己写一个"生成"循环（不许用 model.generate！）
# ---------------------------------------------------------------
# TODO 2: 每次取概率最高的那个词，拼到输入后面，再喂回模型。(Now it is random choice with weight)
#         重复 20 次，打印生成的完整文本。
#
#   会用到的 API（除了上面那些）:
#     torch.argmax(t, dim=-1)      最大值所在的下标
#     torch.cat([a, b], dim=1)     沿第 1 维（序列维度）拼接两个 tensor，注意 a、b 的维度数要一样
with torch.no_grad():
    for _ in range(20):
        next_token = torch.multinomial(soft_vec, num_samples=1)
        print(f"next token is: `{tok.decode(next_token[0])}`")
        next_token = next_token.unsqueeze(0)
        token_ids = torch.cat([token_ids, next_token], dim=1)
        logits = model(token_ids).logits
        soft_vec = torch.softmax(logits[0, -1] / TEMPERATURE, dim=-1)

print(f"The final completed sentence is: `{tok.decode(token_ids)[0]}`")
