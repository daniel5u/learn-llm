"""
把 Hugging Face 上 GPT-2 的官方权重装进我们自己的 GPT（Step 03-05 测试里的里程碑用的就是这套映射）。

这个文件不是练习，直接用就行。两个要注意的差别：
    1. GPT-2 把 q、k、v 三个投影合并存成一个 c_attn: [d_model, 3*d_model]，要切成三份
    2. GPT-2 的线性层是 Conv1D，计算方式是 x @ W（nn.Linear 是 x @ W.T），所以要转置
"""

import torch

from learn_llm.gpt import GPT, GPTConfig


@torch.no_grad()
def load_gpt2(name: str = "openai-community/gpt2"):
    """返回 (model, tokenizer)。model 已经是 eval 模式。"""
    from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer

    hf = GPT2LMHeadModel.from_pretrained(name)
    tok = GPT2Tokenizer.from_pretrained(name)
    c = hf.config
    model = GPT(GPTConfig(c.vocab_size, c.n_positions, c.n_embd, c.n_head, c.n_layer))

    sd = hf.transformer
    model.embed.wte.copy_(sd.wte.weight)
    model.embed.wpe.copy_(sd.wpe.weight)
    for ours, theirs in zip(model.blocks, sd.h):
        ours.ln_1.load_state_dict(theirs.ln_1.state_dict())
        ours.ln_2.load_state_dict(theirs.ln_2.state_dict())

        wq, wk, wv = theirs.attn.c_attn.weight.split(c.n_embd, dim=1)
        bq, bk, bv = theirs.attn.c_attn.bias.split(c.n_embd)
        for proj, w, b in ((ours.attn.q_proj, wq, bq), (ours.attn.k_proj, wk, bk), (ours.attn.v_proj, wv, bv)):
            proj.weight.copy_(w.T)
            proj.bias.copy_(b)
        ours.attn.out_proj.weight.copy_(theirs.attn.c_proj.weight.T)
        ours.attn.out_proj.bias.copy_(theirs.attn.c_proj.bias)

        ours.mlp.fc.weight.copy_(theirs.mlp.c_fc.weight.T)
        ours.mlp.fc.bias.copy_(theirs.mlp.c_fc.bias)
        ours.mlp.proj.weight.copy_(theirs.mlp.c_proj.weight.T)
        ours.mlp.proj.bias.copy_(theirs.mlp.c_proj.bias)
    model.ln_f.load_state_dict(sd.ln_f.state_dict())

    return model.eval(), tok
