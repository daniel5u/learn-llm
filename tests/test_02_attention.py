import math

import torch
import torch.nn.functional as F

from learn_llm.attention import SingleHeadAttention, causal_attention, causal_attention_weights


def rand_qkv(*shape):
    torch.manual_seed(0)
    return torch.randn(*shape), torch.randn(*shape), torch.randn(*shape)


# ---------------------------------------------------------------
# TODO 1: 注意力权重
# ---------------------------------------------------------------
def test_weights_shape():
    q, k, _ = rand_qkv(2, 5, 8)
    assert causal_attention_weights(q, k).shape == (2, 5, 5)


def test_weights_rows_sum_to_one():
    q, k, _ = rand_qkv(2, 5, 8)
    w = causal_attention_weights(q, k)
    torch.testing.assert_close(w.sum(dim=-1), torch.ones(2, 5))


def test_weights_cannot_see_the_future():
    q, k, _ = rand_qkv(2, 5, 8)
    w = causal_attention_weights(q, k)
    future = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    assert (w[:, future] == 0).all(), "j > i 的位置权重必须是 0"
    # 第 0 个 token 只能看到自己
    torch.testing.assert_close(w[:, 0, 0], torch.ones(2))


def test_weights_match_the_formula():
    q, k, _ = rand_qkv(1, 4, 8)
    w = causal_attention_weights(q, k)
    # 手算第 2 个 token 的权重: 只和位置 0、1、2 打分，且要除以 sqrt(d)
    scores = torch.stack([q[0, 2] @ k[0, j] for j in range(3)]) / math.sqrt(8)
    torch.testing.assert_close(w[0, 2, :3], torch.softmax(scores, dim=-1))


# ---------------------------------------------------------------
# TODO 2: 注意力输出
# ---------------------------------------------------------------
def test_attention_matches_pytorch():
    q, k, v = rand_qkv(2, 7, 16)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(causal_attention(q, k, v), expected)


def test_first_token_output_is_its_own_value():
    q, k, v = rand_qkv(2, 5, 8)
    torch.testing.assert_close(causal_attention(q, k, v)[:, 0], v[:, 0])


def test_changing_the_future_does_not_change_the_past():
    q, k, v = rand_qkv(1, 6, 8)
    out1 = causal_attention(q, k, v)
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    q2[:, 4:], k2[:, 4:], v2[:, 4:] = 100.0, -100.0, 42.0  # 乱改最后两个位置
    out2 = causal_attention(q2, k2, v2)
    torch.testing.assert_close(out1[:, :4], out2[:, :4])


def test_supports_extra_head_dimension():
    # [B, H, T, d]：为 Step 03 的多头做准备
    q, k, v = rand_qkv(2, 3, 6, 8)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(causal_attention(q, k, v), expected)


# ---------------------------------------------------------------
# TODO 3 & 4: 单头注意力层
# ---------------------------------------------------------------
def test_layer_shapes():
    attn = SingleHeadAttention(d_model=32, d_head=8)
    assert attn.q_proj.weight.shape == (8, 32)
    assert attn.out_proj.weight.shape == (32, 8)
    assert attn(torch.randn(2, 5, 32)).shape == (2, 5, 32)


def test_layer_uses_projections():
    torch.manual_seed(0)
    attn = SingleHeadAttention(d_model=32, d_head=8)
    x = torch.randn(2, 5, 32)
    expected = attn.out_proj(causal_attention(attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)))
    torch.testing.assert_close(attn(x), expected)


def test_layer_is_causal():
    torch.manual_seed(0)
    attn = SingleHeadAttention(d_model=32, d_head=8)
    x = torch.randn(1, 6, 32)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(1, 3, 32)
    torch.testing.assert_close(attn(x)[:, :3], attn(x2)[:, :3])
