import math

import pytest
import torch

from learn_llm.generate import generate, sample_next_token, top_k_filter
from learn_llm.gpt import GPT, GPTConfig

TINY = GPTConfig(vocab_size=100, max_seq_len=16, d_model=32, n_heads=4, n_layers=2)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(TINY).eval()


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# ---------------------------------------------------------------
# TODO 1: top-k 过滤
# ---------------------------------------------------------------
def test_top_k_keeps_exactly_k_per_row():
    torch.manual_seed(0)
    logits = torch.randn(3, 50)
    out = top_k_filter(logits, 5)
    assert out.shape == (3, 50)
    assert (torch.isfinite(out).sum(dim=-1) == 5).all()


def test_top_k_keeps_the_largest_values_unchanged():
    logits = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0]])
    out = top_k_filter(logits, 2)
    inf = float("-inf")
    torch.testing.assert_close(out, torch.tensor([[inf, 5.0, inf, 4.0, inf]]))


def test_top_k_does_not_modify_its_input():
    logits = torch.randn(2, 10)
    before = logits.clone()
    top_k_filter(logits, 3)
    torch.testing.assert_close(logits, before)


# ---------------------------------------------------------------
# TODO 2: 抽样
# ---------------------------------------------------------------
def test_temperature_zero_is_greedy():
    torch.manual_seed(0)
    logits = torch.randn(4, 100)
    out = sample_next_token(logits, temperature=0.0)
    assert out.shape == (4, 1)
    assert out.dtype == torch.long
    torch.testing.assert_close(out, logits.argmax(dim=-1, keepdim=True))


def test_top_k_1_is_greedy_whatever_the_temperature():
    torch.manual_seed(0)
    logits = torch.randn(4, 100)
    out = sample_next_token(logits, temperature=5.0, top_k=1, generator=gen(0))
    torch.testing.assert_close(out, logits.argmax(dim=-1, keepdim=True))


def test_samples_follow_softmax_probabilities():
    # 概率是 1/7, 2/7, 4/7
    logits = torch.tensor([0.0, math.log(2), math.log(4)]).expand(20000, 3)
    out = sample_next_token(logits, temperature=1.0, generator=gen(0)).squeeze(-1)
    freq = torch.bincount(out, minlength=3).float() / 20000
    torch.testing.assert_close(freq, torch.tensor([1 / 7, 2 / 7, 4 / 7]), atol=0.02, rtol=0)


def test_temperature_changes_how_spread_out_the_samples_are():
    logits = torch.tensor([0.0, 1.0, 2.0]).expand(20000, 3)

    def freq_of_best(t):
        out = sample_next_token(logits, temperature=t, generator=gen(0))
        return (out == 2).float().mean().item()

    # 原始概率下最好的那个约占 67%
    assert freq_of_best(0.2) > 0.95  # 温度低：几乎总选最好的
    assert freq_of_best(100.0) < 0.36  # 温度高：接近均匀，各约 1/3


def test_never_samples_outside_top_k():
    torch.manual_seed(0)
    logits = torch.randn(1, 100).expand(5000, 100)
    allowed = logits[0].topk(10).indices
    out = sample_next_token(logits, temperature=1.0, top_k=10, generator=gen(0))
    assert torch.isin(out, allowed).all()


def test_same_seed_same_samples():
    logits = torch.randn(8, 100)
    a = sample_next_token(logits, temperature=1.0, generator=gen(42))
    b = sample_next_token(logits, temperature=1.0, generator=gen(42))
    torch.testing.assert_close(a, b)


# ---------------------------------------------------------------
# TODO 3: 生成循环
# ---------------------------------------------------------------
def test_generate_shape_and_keeps_the_prompt(model):
    prompt = torch.randint(0, 100, (2, 5))
    out = generate(model, prompt, max_new_tokens=7, generator=gen(0))
    assert out.shape == (2, 12)
    torch.testing.assert_close(out[:, :5], prompt)


def test_greedy_generate_matches_a_manual_loop(model):
    ids = torch.randint(0, 100, (2, 4))
    out = generate(model, ids, max_new_tokens=5, temperature=0.0)
    with torch.no_grad():
        for _ in range(5):
            next_id = model(ids)[:, -1].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
    torch.testing.assert_close(out, ids)


def test_can_generate_past_max_seq_len(model):
    """max_seq_len = 16，但可以一直生成下去：只要每次只喂最后 16 个 token。"""
    prompt = torch.randint(0, 100, (1, 10))
    out = generate(model, prompt, max_new_tokens=20, temperature=0.0)
    assert out.shape == (1, 30)
    # 最后一个 token 只由它前面 16 个 token 决定
    with torch.no_grad():
        expected_last = model(out[:, -17:-1])[:, -1].argmax(dim=-1)
    assert out[0, -1] == expected_last[0]


def test_generate_does_not_build_a_graph(model):
    out = generate(model, torch.randint(0, 100, (1, 3)), max_new_tokens=2, generator=gen(0))
    assert not out.requires_grad


# ---------------------------------------------------------------
# 里程碑：用真正的 GPT-2 写句子，和 Hugging Face 的 generate 逐 token 一致
# ---------------------------------------------------------------
@pytest.fixture(scope="module")
def gpt2():
    from transformers.models.gpt2 import GPT2LMHeadModel

    from learn_llm.pretrained import load_gpt2

    model, tok = load_gpt2()
    hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    return model, hf, tok


def test_greedy_matches_huggingface(gpt2):
    model, hf, tok = gpt2
    ids = torch.tensor([tok.encode("The Eiffel Tower is located in the city of")])
    ours = generate(model, ids, max_new_tokens=15, temperature=0.0)
    with torch.no_grad():
        expected = hf.generate(ids, max_new_tokens=15, do_sample=False, pad_token_id=tok.eos_token_id)
    assert tok.decode(ours[0]) == tok.decode(expected[0])


def test_sampling_is_reproducible(gpt2):
    model, _, tok = gpt2
    ids = torch.tensor([tok.encode("Once upon a time")])
    a = generate(model, ids, max_new_tokens=10, temperature=0.8, top_k=50, generator=gen(123))
    b = generate(model, ids, max_new_tokens=10, temperature=0.8, top_k=50, generator=gen(123))
    torch.testing.assert_close(a, b)
