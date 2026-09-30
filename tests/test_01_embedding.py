import pytest
import torch
from torch import nn

from learn_llm.embedding import Embedding


def test_parameters_have_right_shapes():
    emb = Embedding(vocab_size=100, max_seq_len=16, d_model=8)
    assert isinstance(emb.wte, nn.Parameter)
    assert isinstance(emb.wpe, nn.Parameter)
    assert emb.wte.shape == (100, 8)
    assert emb.wpe.shape == (16, 8)
    # 两张表都要算作可学习参数
    assert sum(p.numel() for p in emb.parameters()) == 100 * 8 + 16 * 8


def test_does_not_use_nn_embedding():
    emb = Embedding(vocab_size=100, max_seq_len=16, d_model=8)
    assert not any(isinstance(m, nn.Embedding) for m in emb.modules()), "自己实现查表，不许用 nn.Embedding"


def test_output_shape():
    emb = Embedding(vocab_size=100, max_seq_len=16, d_model=8)
    token_ids = torch.randint(0, 100, (3, 5))
    assert emb(token_ids).shape == (3, 5, 8)


def test_each_position_is_token_vector_plus_position_vector():
    emb = Embedding(vocab_size=100, max_seq_len=16, d_model=8)
    token_ids = torch.randint(0, 100, (2, 6))
    out = emb(token_ids)
    for b in range(2):
        for t in range(6):
            expected = emb.wte[token_ids[b, t]] + emb.wpe[t]
            torch.testing.assert_close(out[b, t], expected)


def test_same_token_at_different_positions_gets_different_vectors():
    emb = Embedding(vocab_size=100, max_seq_len=16, d_model=8)
    out = emb(torch.tensor([[7, 7, 7]]))
    assert not torch.allclose(out[0, 0], out[0, 1])
    # 同一个 token，差别只来自位置向量
    torch.testing.assert_close(out[0, 1] - out[0, 0], emb.wpe[1] - emb.wpe[0])


def test_works_with_shorter_sequences_than_max():
    emb = Embedding(vocab_size=100, max_seq_len=16, d_model=8)
    assert emb(torch.tensor([[1]])).shape == (1, 1, 8)
    assert emb(torch.randint(0, 100, (1, 16))).shape == (1, 16, 8)


# ---------------------------------------------------------------
# 里程碑：加载 GPT-2 官方权重，和 Hugging Face 的结果对比
# ---------------------------------------------------------------
@pytest.fixture(scope="module")
def hf_gpt2():
    from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer

    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    tok = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    return model, tok


def test_matches_huggingface_gpt2(hf_gpt2):
    hf, tok = hf_gpt2
    cfg = hf.config
    emb = Embedding(vocab_size=cfg.vocab_size, max_seq_len=cfg.n_positions, d_model=cfg.n_embd)
    with torch.no_grad():
        emb.wte.copy_(hf.transformer.wte.weight)
        emb.wpe.copy_(hf.transformer.wpe.weight)

    token_ids = torch.tensor([tok.encode("Hello, my dog is cute and my cat is lazy.")])
    with torch.no_grad():
        # hidden_states[0] 就是 HF 的 GPT-2 在进入第一个 Transformer block 之前的向量
        expected = hf(token_ids, output_hidden_states=True).hidden_states[0]
        ours = emb(token_ids)
    torch.testing.assert_close(ours, expected)
