import pytest
import torch
import torch.nn.functional as F
from torch import nn

from learn_llm.attention import MultiHeadAttention
from learn_llm.block import MLP, Block


# ---------------------------------------------------------------
# TODO 1 & 2: MLP
# ---------------------------------------------------------------
@pytest.fixture
def mlp():
    torch.manual_seed(0)
    return MLP(d_model=32)


def test_mlp_init(mlp):
    assert mlp.fc.weight.shape == (128, 32)
    assert mlp.proj.weight.shape == (32, 128)
    assert isinstance(mlp.gelu, nn.GELU)
    assert mlp.gelu.approximate == "tanh"


def test_mlp_forward(mlp):
    x = torch.randn(2, 5, 32)
    out = mlp(x)
    assert out.shape == (2, 5, 32)
    h = x @ mlp.fc.weight.T + mlp.fc.bias  # [2, 5, 128]
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
# TODO 3 & 4: Block
# ---------------------------------------------------------------
@pytest.fixture
def block():
    torch.manual_seed(0)
    return Block(d_model=32, n_heads=4)


def test_block_init(block):
    assert isinstance(block.ln_1, nn.LayerNorm)
    assert isinstance(block.attn, MultiHeadAttention)
    assert isinstance(block.ln_2, nn.LayerNorm)
    assert isinstance(block.mlp, MLP)
    assert block.ln_1.normalized_shape == (32,)
    assert block.attn.n_heads == 4


def test_block_forward_shape(block):
    assert block(torch.randn(2, 5, 32)).shape == (2, 5, 32)


def test_block_is_causal(block):
    x = torch.randn(1, 6, 32)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(1, 3, 32)
    torch.testing.assert_close(block(x)[:, :3], block(x2)[:, :3])


def test_block_does_not_modify_its_input(block):
    """x = x + ... 会新建一个 tensor；x += ... 是原地修改，会把调用者传进来的 x 也改掉。"""
    x = torch.randn(2, 5, 32)
    x_before = x.clone()
    block(x)
    torch.testing.assert_close(x, x_before)


def test_block_backward_works(block):
    """原地修改还会破坏反向传播：ln_1 为了算梯度存下了 x，x += ... 把它覆盖了。"""
    x = torch.randn(2, 5, 32, requires_grad=True)
    block(x * 1).sum().backward()
    assert x.grad is not None


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
    ours.ln_1.load_state_dict(hf_block.ln_1.state_dict())
    ours.ln_2.load_state_dict(hf_block.ln_2.state_dict())
    with torch.no_grad():
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
