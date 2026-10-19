import math

import pytest
import torch
import torch.nn.functional as F

from learn_llm.attention import MultiHeadAttention
from learn_llm.block import MLP, Block, LayerNorm, gelu


# ---------------------------------------------------------------
# TODO 1 & 2: LayerNorm
# ---------------------------------------------------------------
def test_layernorm_init():
    ln = LayerNorm(16)
    assert isinstance(ln.weight, torch.nn.Parameter)
    assert isinstance(ln.bias, torch.nn.Parameter)
    torch.testing.assert_close(ln.weight.data, torch.ones(16))
    torch.testing.assert_close(ln.bias.data, torch.zeros(16))
    assert ln.eps == 1e-5


def test_layernorm_output_has_mean_0_and_var_1():
    torch.manual_seed(0)
    ln = LayerNorm(16)
    x = torch.randn(2, 5, 16) * 7 + 3
    y = ln(x)
    assert y.shape == (2, 5, 16)
    # 每个 token 自己的 16 个数：均值 0，方差 1
    torch.testing.assert_close(y.mean(dim=-1), torch.zeros(2, 5), atol=1e-5, rtol=0)
    torch.testing.assert_close(y.var(dim=-1, unbiased=False), torch.ones(2, 5), atol=1e-3, rtol=0)


def test_layernorm_matches_pytorch():
    torch.manual_seed(0)
    ln = LayerNorm(16)
    with torch.no_grad():
        ln.weight.copy_(torch.randn(16))
        ln.bias.copy_(torch.randn(16))
    x = torch.randn(2, 5, 16)
    expected = F.layer_norm(x, (16,), ln.weight, ln.bias, eps=1e-5)
    torch.testing.assert_close(ln(x), expected)


def test_layernorm_treats_each_token_independently():
    torch.manual_seed(0)
    ln = LayerNorm(16)
    x = torch.randn(2, 5, 16)
    x2 = x.clone()
    x2[1, 3] = torch.randn(16) * 100
    y, y2 = ln(x), ln(x2)
    changed = torch.zeros(2, 5, dtype=torch.bool)
    changed[1, 3] = True
    torch.testing.assert_close(y[~changed], y2[~changed])


def test_layernorm_constant_input_does_not_blow_up():
    # 所有数都一样时方差是 0，靠 eps 避免除以 0
    y = LayerNorm(16)(torch.full((1, 1, 16), 5.0))
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, torch.zeros(1, 1, 16))


# ---------------------------------------------------------------
# TODO 3: gelu
# ---------------------------------------------------------------
def test_gelu_matches_pytorch():
    x = torch.linspace(-6, 6, 101)
    torch.testing.assert_close(gelu(x), F.gelu(x, approximate="tanh"))


def test_gelu_shape_of_the_curve():
    torch.testing.assert_close(gelu(torch.tensor(0.0)), torch.tensor(0.0))
    # 很大的正数 ≈ 自己，很负的数 ≈ 0
    torch.testing.assert_close(gelu(torch.tensor(10.0)), torch.tensor(10.0))
    assert gelu(torch.tensor(-10.0)).abs() < 1e-6


# ---------------------------------------------------------------
# TODO 4 & 5: MLP
# ---------------------------------------------------------------
@pytest.fixture
def mlp():
    torch.manual_seed(0)
    return MLP(d_model=32)


def test_mlp_init(mlp):
    assert mlp.fc.weight.shape == (128, 32)
    assert mlp.proj.weight.shape == (32, 128)


def test_mlp_forward(mlp):
    x = torch.randn(2, 5, 32)
    out = mlp(x)
    assert out.shape == (2, 5, 32)
    h = x @ mlp.fc.weight.T + mlp.fc.bias
    h = F.gelu(h, approximate="tanh")
    torch.testing.assert_close(out, h @ mlp.proj.weight.T + mlp.proj.bias)


def test_mlp_tokens_do_not_talk_to_each_other(mlp):
    """改动第 3 个 token，前面和后面的 token 都不受影响（attention 里只有"后面"会受影响）。"""
    x = torch.randn(1, 6, 32)
    x2 = x.clone()
    x2[:, 3] = torch.randn(32)
    y, y2 = mlp(x), mlp(x2)
    others = [0, 1, 2, 4, 5]
    torch.testing.assert_close(y[:, others], y2[:, others])


# ---------------------------------------------------------------
# TODO 6 & 7: Block
# ---------------------------------------------------------------
@pytest.fixture
def block():
    torch.manual_seed(0)
    return Block(d_model=32, n_heads=4)


def test_block_init(block):
    assert isinstance(block.ln_1, LayerNorm)
    assert isinstance(block.attn, MultiHeadAttention)
    assert isinstance(block.ln_2, LayerNorm)
    assert isinstance(block.mlp, MLP)
    assert block.attn.n_heads == 4


def test_block_forward_shape(block):
    assert block(torch.randn(2, 5, 32)).shape == (2, 5, 32)


def test_block_is_causal(block):
    x = torch.randn(1, 6, 32)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(1, 3, 32)
    torch.testing.assert_close(block(x)[:, :3], block(x2)[:, :3])


def test_block_with_silent_sublayers_is_identity(block):
    """两个子层的输出投影都清零 -> 子层输出恒为 0 -> 残差流原样通过。"""
    with torch.no_grad():
        for layer in (block.attn.out_proj, block.mlp.proj):
            layer.weight.zero_()
            layer.bias.zero_()
    x = torch.randn(2, 5, 32)
    torch.testing.assert_close(block(x), x)


def test_block_is_pre_ln(block):
    with torch.no_grad():
        for ln in (block.ln_1, block.ln_2):
            ln.weight.copy_(torch.randn(32))
            ln.bias.copy_(torch.randn(32))
    x = torch.randn(2, 5, 32) * 3
    h = x + block.attn(block.ln_1(x))
    expected = h + block.mlp(block.ln_2(h))
    torch.testing.assert_close(block(x), expected)


# ---------------------------------------------------------------
# 里程碑：加载 GPT-2 完整的第 0 层，和 Hugging Face 对比
# ---------------------------------------------------------------
@pytest.fixture(scope="module")
def hf_gpt2():
    from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer

    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    tok = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    return model, tok


def load_gpt2_block(ours: Block, hf_block) -> None:
    """GPT-2 的线性层都是转置着存的（x @ W），所以拷进 nn.Linear 时要 .T；LayerNorm 直接拷。"""
    from test_03_multihead import load_gpt2_attention

    load_gpt2_attention(ours.attn, hf_block.attn)
    with torch.no_grad():
        for name in ("ln_1", "ln_2"):
            getattr(ours, name).weight.copy_(getattr(hf_block, name).weight)
            getattr(ours, name).bias.copy_(getattr(hf_block, name).bias)
        ours.mlp.fc.weight.copy_(hf_block.mlp.c_fc.weight.T)
        ours.mlp.fc.bias.copy_(hf_block.mlp.c_fc.bias)
        ours.mlp.proj.weight.copy_(hf_block.mlp.c_proj.weight.T)
        ours.mlp.proj.bias.copy_(hf_block.mlp.c_proj.bias)


def test_matches_huggingface_gpt2(hf_gpt2):
    hf, tok = hf_gpt2
    block = Block(d_model=hf.config.n_embd, n_heads=hf.config.n_head)
    load_gpt2_block(block, hf.transformer.h[0])

    token_ids = torch.tensor([tok.encode("Hello, my dog is cute and my cat is lazy.")])
    with torch.no_grad():
        hidden = hf(token_ids, output_hidden_states=True).hidden_states
        # hidden[0] 是 embedding 的输出（第 0 层的输入），hidden[1] 是第 0 层的输出
        ours = block(hidden[0])
    torch.testing.assert_close(ours, hidden[1], atol=1e-4, rtol=1e-4)
