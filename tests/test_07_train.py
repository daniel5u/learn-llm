import math

import pytest
import torch

from learn_llm.gpt import GPT, GPTConfig
from learn_llm.train import TrainConfig, compute_loss, estimate_loss, get_batch, train, train_step

TINY = GPTConfig(vocab_size=20, max_seq_len=16, d_model=32, n_heads=4, n_layers=2)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(TINY)


def gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# ---------------------------------------------------------------
# TODO 1: get_batch
# ---------------------------------------------------------------
def test_batch_shapes():
    data = torch.randint(0, 20, (1000,))
    x, y = get_batch(data, batch_size=4, seq_len=8, generator=gen())
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert x.dtype == torch.long and y.dtype == torch.long


def test_batch_rows_are_contiguous_windows_and_y_is_x_shifted():
    # data[i] == i，所以每个窗口都应该是连续的整数
    data = torch.arange(1000)
    x, y = get_batch(data, batch_size=16, seq_len=8, generator=gen())
    for row_x, row_y in zip(x, y):
        start = row_x[0].item()
        torch.testing.assert_close(row_x, torch.arange(start, start + 8))
        torch.testing.assert_close(row_y, torch.arange(start + 1, start + 9))


def test_batch_never_goes_out_of_bounds():
    data = torch.arange(10)
    for seed in range(50):
        x, y = get_batch(data, batch_size=8, seq_len=9, generator=gen(seed))
        assert y.max() <= 9  # seq_len = N - 1 时，唯一合法的起点是 0


def test_batch_is_random_but_reproducible():
    data = torch.arange(1000)
    a, _ = get_batch(data, 8, 8, generator=gen(1))
    b, _ = get_batch(data, 8, 8, generator=gen(1))
    c, _ = get_batch(data, 8, 8, generator=gen(2))
    torch.testing.assert_close(a, b)
    assert not torch.equal(a, c)


# ---------------------------------------------------------------
# TODO 2: compute_loss
# ---------------------------------------------------------------
def test_loss_is_a_scalar():
    loss = compute_loss(torch.randn(2, 5, 20), torch.randint(0, 20, (2, 5)))
    assert loss.shape == ()


def test_loss_matches_the_definition():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 20)
    targets = torch.randint(0, 20, (2, 5))
    log_probs = torch.log_softmax(logits, dim=-1)
    # 每个位置：正确答案的 -log 概率；再对 2*5 个位置求平均
    expected = -torch.stack([log_probs[b, t, targets[b, t]] for b in range(2) for t in range(5)]).mean()
    torch.testing.assert_close(compute_loss(logits, targets), expected)


def test_loss_of_uniform_guess_is_ln_vocab():
    loss = compute_loss(torch.zeros(2, 5, 20), torch.randint(0, 20, (2, 5)))
    torch.testing.assert_close(loss, torch.tensor(math.log(20)))


def test_untrained_model_is_close_to_uniform_guess(model):
    """刚初始化的模型应该"什么都不知道"。如果这里 loss 远大于 ln(V)，说明初始化有问题。"""
    data = torch.randint(0, 20, (1000,))
    x, y = get_batch(data, 8, 16, generator=gen())
    loss = compute_loss(model(x), y).item()
    assert abs(loss - math.log(20)) < 0.1


# ---------------------------------------------------------------
# TODO 3: train_step
# ---------------------------------------------------------------
def test_train_step_returns_float_and_updates_parameters(model):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [p.clone() for p in model.parameters()]
    x, y = get_batch(torch.randint(0, 20, (1000,)), 4, 16, generator=gen())
    loss = train_step(model, opt, x, y)
    assert isinstance(loss, float)
    assert all(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_train_step_clears_old_gradients(model):
    """同一个 batch 跑两次 forward/backward：梯度应该一样，而不是第二次变成两倍。"""
    opt = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0：参数不动，只看梯度
    x, y = get_batch(torch.randint(0, 20, (1000,)), 4, 16, generator=gen())
    train_step(model, opt, x, y)
    first = model.embed.wte.grad.clone()
    train_step(model, opt, x, y)
    torch.testing.assert_close(model.embed.wte.grad, first)


def test_can_overfit_a_single_batch(model):
    """训练代码最经典的检查：反复喂同一个 batch，模型应该能把它"背"下来，loss 降到接近 0。
    做不到的话，训练循环里一定有 bug。"""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x, y = get_batch(torch.randint(0, 20, (1000,)), 4, 16, generator=gen())
    losses = [train_step(model, opt, x, y) for _ in range(150)]
    assert losses[0] > 2.5
    assert losses[-1] < 0.1


# ---------------------------------------------------------------
# TODO 4: estimate_loss
# ---------------------------------------------------------------
def test_estimate_loss_averages_and_restores_train_mode(model):
    data = torch.randint(0, 20, (1000,))
    model.train()
    loss = estimate_loss(model, data, batch_size=4, seq_len=16, n_batches=5, generator=gen())
    assert isinstance(loss, float)
    assert model.training, "评估完要切回 train 模式"
    assert all(p.grad is None for p in model.parameters()), "评估不应该产生梯度"


def test_estimate_loss_is_the_mean_over_batches(model):
    data = torch.randint(0, 20, (1000,))
    loss = estimate_loss(model, data, batch_size=4, seq_len=16, n_batches=3, generator=gen(7))
    g = gen(7)
    with torch.no_grad():
        expected = sum(compute_loss(model(x), y).item() for x, y in (get_batch(data, 4, 16, g) for _ in range(3))) / 3
    assert loss == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------
# TODO 5: train
# ---------------------------------------------------------------
def test_train_history_format(model):
    data = torch.randint(0, 20, (500,))
    cfg = TrainConfig(batch_size=4, seq_len=16, max_steps=25, eval_interval=10, eval_batches=2)
    history = train(model, data, data, cfg)
    assert [h["step"] for h in history] == [0, 10, 20, 25]
    assert all(set(h) == {"step", "train", "val"} for h in history)


def test_train_learns_a_pattern(model):
    """数据是 0,1,...,6,0,1,...,6,... 的循环：看过一个数字就能确定下一个。
    训练前 loss ≈ ln(20) ≈ 3.0，训练后应该接近 0，而且 val 也要降（这是规律，不是死记）。"""
    data = torch.arange(3000) % 7
    cfg = TrainConfig(batch_size=8, seq_len=16, lr=3e-3, max_steps=150, eval_interval=50, eval_batches=4)
    history = train(model, data[:2500], data[2500:], cfg)
    assert history[0]["val"] > 2.5
    assert history[-1]["train"] < 0.1
    assert history[-1]["val"] < 0.1
