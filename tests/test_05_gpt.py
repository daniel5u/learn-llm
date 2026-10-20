import pytest
import torch
from torch import nn

from learn_llm.block import Block
from learn_llm.embedding import Embedding
from learn_llm.gpt import GPT, GPTConfig

TINY = GPTConfig(vocab_size=100, max_seq_len=16, d_model=32, n_heads=4, n_layers=3)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(TINY)


# ---------------------------------------------------------------
# TODO 1: 初始化
# ---------------------------------------------------------------
def test_init(model):
    assert isinstance(model.embed, Embedding)
    assert isinstance(model.blocks, nn.ModuleList), "要用 nn.ModuleList，普通 list 里的参数不会被注册"
    assert len(model.blocks) == 3
    assert all(isinstance(b, Block) for b in model.blocks)
    assert isinstance(model.ln_f, nn.LayerNorm)


def test_blocks_are_independent(model):
    """每个 block 有自己的参数，不是同一个 Block 被引用了 3 次。"""
    assert model.blocks[0] is not model.blocks[1]
    assert model.blocks[0].mlp.fc.weight is not model.blocks[1].mlp.fc.weight


def test_no_separate_output_layer(model):
    """权重共享：整个模型里只有一个 [vocab_size, d_model] 的参数，就是 wte。"""
    vocab_shaped = [n for n, p in model.named_parameters() if p.shape == (100, 32)]
    assert vocab_shaped == ["embed.wte"]


def test_gpt2_small_has_124m_parameters():
    # meta device: 只记录形状，不真的分配内存，所以建一个 1.24 亿参数的模型也是瞬间完成
    with torch.device("meta"):
        model = GPT(GPTConfig())
    assert sum(p.numel() for p in model.parameters()) == 124_439_808


# ---------------------------------------------------------------
# TODO 2: forward
# ---------------------------------------------------------------
def test_forward_shape(model):
    token_ids = torch.randint(0, 100, (2, 7))
    assert model(token_ids).shape == (2, 7, 100)


def test_forward_follows_the_pipeline(model):
    token_ids = torch.randint(0, 100, (2, 7))
    x = model.embed(token_ids)
    for block in model.blocks:
        x = block(x)
    expected = model.ln_f(x) @ model.embed.wte.T
    torch.testing.assert_close(model(token_ids), expected)


def test_forward_is_causal(model):
    """位置 t 的预测只依赖 token 0..t，后面的 token 怎么改都不影响它。"""
    a = torch.randint(0, 100, (1, 8))
    b = a.clone()
    b[:, 5:] = torch.randint(0, 100, (1, 3))
    torch.testing.assert_close(model(a)[:, :5], model(b)[:, :5])


def test_wte_gets_gradient_from_both_ends(model):
    """wte 在输入端（查表）和输出端（unembedding）都被用到，两边的梯度都会流回它。"""
    token_ids = torch.tensor([[1, 2, 3]])
    model(token_ids).sum().backward()
    grad = model.embed.wte.grad
    assert grad is not None
    # 没出现在输入里的 token（比如 50）也有梯度 —— 只能是从输出端来的
    assert grad[50].abs().sum() > 0


# ---------------------------------------------------------------
# 里程碑：加载 GPT-2 全部权重，logits 和 Hugging Face 一致，并且能"说对话"
# ---------------------------------------------------------------
@pytest.fixture(scope="module")
def gpt2_pair():
    from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer

    from test_04_block import load_gpt2_block

    hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()
    tok = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

    ours = GPT(GPTConfig()).eval()
    with torch.no_grad():
        ours.embed.wte.copy_(hf.transformer.wte.weight)
        ours.embed.wpe.copy_(hf.transformer.wpe.weight)
        for our_block, hf_block in zip(ours.blocks, hf.transformer.h):
            load_gpt2_block(our_block, hf_block)
        ours.ln_f.load_state_dict(hf.transformer.ln_f.state_dict())
    return ours, hf, tok


def test_matches_huggingface_gpt2(gpt2_pair):
    ours, hf, tok = gpt2_pair
    token_ids = torch.tensor([tok.encode("Hello, my dog is cute and my cat is lazy.")])
    with torch.no_grad():
        expected = hf(token_ids).logits
        logits = ours(token_ids)
    # 12 层叠下来误差会累积一点，所以容差比前面的步骤宽
    torch.testing.assert_close(logits, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "prompt, answer",
    [
        ("The Eiffel Tower is located in the city of", " Paris"),
        ("One, two, three, four,", " five"),
        ("Monday, Tuesday, Wednesday,", " Thursday"),
    ],
)
def test_predicts_the_next_token(gpt2_pair, prompt, answer):
    ours, _, tok = gpt2_pair
    token_ids = torch.tensor([tok.encode(prompt)])
    with torch.no_grad():
        logits = ours(token_ids)
    # 只看最后一个位置：它看过了整句话，预测的是句子后面的那个 token
    next_id = logits[0, -1].argmax().item()
    assert tok.decode(next_id) == answer
