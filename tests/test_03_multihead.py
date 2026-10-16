import pytest
import torch

from learn_llm.attention import MultiHeadAttention, causal_attention, causal_attention_weights


@pytest.fixture
def attn():
    torch.manual_seed(0)
    return MultiHeadAttention(d_model=32, n_heads=4)


# ---------------------------------------------------------------
# TODO 1: 初始化
# ---------------------------------------------------------------
def test_init(attn):
    assert attn.n_heads == 4
    assert attn.d_head == 8
    for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj):
        assert proj.weight.shape == (32, 32)


def test_d_model_must_be_divisible_by_n_heads():
    with pytest.raises(AssertionError):
        MultiHeadAttention(d_model=32, n_heads=5)


# ---------------------------------------------------------------
# TODO 2 & 3: split / merge
# ---------------------------------------------------------------
def test_split_heads_shape(attn):
    assert attn.split_heads(torch.randn(2, 5, 32)).shape == (2, 4, 5, 8)


def test_split_heads_gives_each_head_a_consecutive_slice(attn):
    x = torch.randn(2, 5, 32)
    heads = attn.split_heads(x)
    for h in range(4):
        # 第 h 个头拿到的是最后一维的第 [8h, 8h+8) 段
        torch.testing.assert_close(heads[:, h], x[:, :, 8 * h : 8 * (h + 1)])


def test_merge_heads_undoes_split_heads(attn):
    x = torch.randn(2, 5, 32)
    merged = attn.merge_heads(attn.split_heads(x))
    assert merged.shape == (2, 5, 32)
    torch.testing.assert_close(merged, x)


# ---------------------------------------------------------------
# TODO 4: forward
# ---------------------------------------------------------------
def test_forward_shape(attn):
    assert attn(torch.randn(2, 5, 32)).shape == (2, 5, 32)


def test_forward_is_causal(attn):
    x = torch.randn(1, 6, 32)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(1, 3, 32)
    torch.testing.assert_close(attn(x)[:, :3], attn(x2)[:, :3])


def test_multihead_is_a_sum_of_independent_heads(attn):
    """多头注意力 = H 个互相独立的小注意力头，各自的输出加在一起。

    每个头都读取完整的 x，只是用的是大矩阵里属于自己的那一段参数。
    """
    x = torch.randn(2, 5, 32)
    total = torch.zeros(2, 5, 32)
    for h in range(4):
        rows = slice(8 * h, 8 * (h + 1))
        # 第 h 个头自己的 W_q / W_k / W_v: [d_head, d_model]，输入是完整的 x
        q = x @ attn.q_proj.weight[rows].T + attn.q_proj.bias[rows]
        k = x @ attn.k_proj.weight[rows].T + attn.k_proj.bias[rows]
        v = x @ attn.v_proj.weight[rows].T + attn.v_proj.bias[rows]
        head_out = causal_attention(q, k, v)  # [B, T, d_head]
        # 第 h 个头自己的 W_o: [d_model, d_head]，把结果写回残差流
        total = total + head_out @ attn.out_proj.weight[:, rows].T
    total = total + attn.out_proj.bias
    torch.testing.assert_close(attn(x), total, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------
# TODO 5: attention pattern
# ---------------------------------------------------------------
def test_attention_pattern(attn):
    x = torch.randn(2, 5, 32)
    pattern = attn.attention_pattern(x)
    assert pattern.shape == (2, 4, 5, 5)
    torch.testing.assert_close(pattern.sum(dim=-1), torch.ones(2, 4, 5))
    expected = causal_attention_weights(attn.split_heads(attn.q_proj(x)), attn.split_heads(attn.k_proj(x)))
    torch.testing.assert_close(pattern, expected)


# ---------------------------------------------------------------
# 里程碑：加载 GPT-2 第 0 层的注意力权重，和 Hugging Face 对比
# ---------------------------------------------------------------
@pytest.fixture(scope="module")
def hf_gpt2():
    from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer

    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    tok = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    return model, tok


def load_gpt2_attention(ours: MultiHeadAttention, hf_attn) -> None:
    """GPT-2 把 q、k、v 三个投影合并存成了一个矩阵 c_attn，而且是转置着存的：

    c_attn.weight: [d_model, 3 * d_model]，计算方式是 x @ W（nn.Linear 是 x @ W.T）
    所以要沿着第 1 维切成三份，再各自转置。
    """
    d = ours.q_proj.weight.shape[0]
    wq, wk, wv = hf_attn.c_attn.weight.split(d, dim=1)
    bq, bk, bv = hf_attn.c_attn.bias.split(d)
    with torch.no_grad():
        ours.q_proj.weight.copy_(wq.T)
        ours.k_proj.weight.copy_(wk.T)
        ours.v_proj.weight.copy_(wv.T)
        ours.q_proj.bias.copy_(bq)
        ours.k_proj.bias.copy_(bk)
        ours.v_proj.bias.copy_(bv)
        ours.out_proj.weight.copy_(hf_attn.c_proj.weight.T)
        ours.out_proj.bias.copy_(hf_attn.c_proj.bias)


def test_matches_huggingface_gpt2(hf_gpt2):
    hf, tok = hf_gpt2
    block = hf.transformer.h[0]
    attn = MultiHeadAttention(d_model=hf.config.n_embd, n_heads=hf.config.n_head)
    load_gpt2_attention(attn, block.attn)

    token_ids = torch.tensor([tok.encode("Hello, my dog is cute and my cat is lazy.")])
    with torch.no_grad():
        # 第 0 层注意力的真实输入：embedding 的输出再过一个 LayerNorm
        x = block.ln_1(hf(token_ids, output_hidden_states=True).hidden_states[0])
        expected = block.attn(x)[0]
        ours = attn(x)
    torch.testing.assert_close(ours, expected, atol=1e-4, rtol=1e-4)
