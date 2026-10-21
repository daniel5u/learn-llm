"""
Step 07: 训练 —— 让模型从随机参数开始，自己学会"预测下一个 token"

一次训练迭代（一个 step）的数据流：

    data [N]                                 整个数据集：一条很长的 token 序列
      │ get_batch: 随机切 B 个长度 T+1 的窗口
      ▼
    x [B, T]   y [B, T]                      y 就是 x 往后挪一位：y[b, t] = "x[b, t] 后面的那个 token"
      │ model
      ▼
    logits [B, T, V]
      │ compute_loss: 交叉熵
      ▼
    loss（一个数）                            "模型给正确答案的概率" 取 -log，再对 B*T 个位置求平均
      │ loss.backward()                      算出每个参数的梯度：往哪个方向改能让 loss 变小
      │ optimizer.step()                     按梯度更新参数
      ▼
    参数变好了一点点  ──> 回到开头，换一批数据再来

几个要点：
    1. 一个窗口里 T 个位置同时在学：位置 t 看过 x[0..t]，要预测 y[t] = x[t+1]。
       因果 mask 保证它没法偷看答案，所以一次 forward 就得到 T 个训练样本。
    2. loss 的"基准线"：模型完全瞎猜（均匀分布）时，loss = ln(V)。
       刚初始化的模型应该接近这个值；训练后要明显低于它，才说明学到了东西。
    3. train loss 和 val loss：val 用的是训练时没见过的数据。
       train 降而 val 不降（甚至升）= 过拟合，模型在"背答案"而不是在学规律。

测试: uv run pytest tests/test_07_train.py -v
试玩: uv run python -m learn_llm.train            （CPU 上几分钟，训练一个写"莎士比亚"的小模型）
"""

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from learn_llm.generate import generate
from learn_llm.gpt import GPT, GPTConfig


def get_batch(
    data: Tensor,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """data [N] -> x [B, T], y [B, T]

    随机选 B 个起点 i，每个起点切出:
        x = data[i     : i + T    ]
        y = data[i + 1 : i + T + 1]     （比 x 往后挪一位）
    """
    # TODO 1: 随机起点 -> 切窗口 -> 叠成 batch
    #
    #   会用到的 API:
    #     torch.randint(low, high, (n,), generator=generator)   [low, high) 里的 n 个随机整数
    #                                  起点最大只能到 N - T - 1，否则 y 会越界
    #     torch.stack([t1, t2, ...])   把 B 个 [T] 叠成 [B, T]
    index = torch.randint(
        low=0,
        high=data.shape[0] - seq_len,
        size=(batch_size,),
        generator=generator,
    )
    x_slices = [data[i : i + seq_len] for i in index]
    y_slices = [data[i + 1 : i + seq_len + 1] for i in index]
    x = torch.stack([x_slice for x_slice in x_slices])
    y = torch.stack([y_slice for y_slice in y_slices])
    return x, y


def compute_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """logits [B, T, V], targets [B, T] -> 标量 loss（所有 B*T 个位置的平均交叉熵）"""
    # TODO 2: 把前两维摊平，交给 F.cross_entropy
    #
    #   会用到的 API:
    #     F.cross_entropy(logits, targets)   要求 logits [N, V]、targets [N]，
    #                                        内部做了 softmax + 取正确答案的 -log 概率 + 求平均
    #     t.reshape(-1, V)                   [B, T, V] -> [B*T, V]
    #     t.reshape(-1)                      [B, T]    -> [B*T]
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    return F.cross_entropy(input=logits, target=targets)


def train_step(
    model: GPT, optimizer: torch.optim.Optimizer, x: Tensor, y: Tensor
) -> float:
    """一次参数更新，返回这一步的 loss（Python float）"""
    # TODO 3: 训练的核心 5 步:
    #           forward -> 算 loss -> 清空旧梯度 -> backward -> 更新参数
    #
    #   会用到的 API:
    #     optimizer.zero_grad()        PyTorch 的梯度是"累加"的，不清空就会和上一步的混在一起
    #     loss.backward()
    #     optimizer.step()
    #     loss.item()                  0 维 tensor -> Python float
    logits = model(x)
    ce = compute_loss(logits, y)
    optimizer.zero_grad()
    ce.backward()
    optimizer.step()
    return ce.item()


@torch.no_grad()
def estimate_loss(
    model: GPT,
    data: Tensor,
    batch_size: int,
    seq_len: int,
    n_batches: int,
    generator: torch.Generator | None = None,
) -> float:
    """在 n_batches 个随机 batch 上求平均 loss。单个 batch 的 loss 噪声很大，平均一下才看得清趋势。"""
    # TODO 4: 切到 eval 模式 -> 循环算 loss 求平均 -> 切回 train 模式
    #
    #   会用到的 API:
    #     model.eval() / model.train()   切换模式（我们的模型里还没有 dropout，但养成习惯）
    #     @torch.no_grad()               已经加在函数上了：只评估、不训练，不需要梯度
    model.eval()
    loss = 0.0
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, seq_len, generator)
        logits = model(x)
        l = compute_loss(logits, y)
        loss += l.item()
    model.train()
    return loss / n_batches


@dataclass
class TrainConfig:
    batch_size: int = 32
    seq_len: int = 128  # 必须 <= 模型的 max_seq_len
    lr: float = 1e-3
    max_steps: int = 2000
    eval_interval: int = 200  # 每多少步评估一次
    eval_batches: int = 20
    seed: int = 0


def train(
    model: GPT, train_data: Tensor, val_data: Tensor, cfg: TrainConfig
) -> list[dict]:
    """训练 cfg.max_steps 步，返回评估记录 [{"step": ..., "train": ..., "val": ...}, ...]

    第 0 步（训练前）和之后每 eval_interval 步、以及最后一步都要评估一次。
    """
    generator = torch.Generator().manual_seed(cfg.seed)
    # TODO 5: 建优化器 -> 循环 max_steps 次 train_step -> 按时评估、记录、打印
    #
    #   会用到的 API:
    #     torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    #     estimate_loss(...)          train 和 val 各估一次，都传入 generator
    #     print(f"step {step}: train {train_loss:.3f}, val {val_loss:.3f}")
    #
    #   提示：for step in range(cfg.max_steps + 1)，在循环开头判断要不要评估，
    #         最后一轮（step == max_steps）只评估、不再训练。
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    history = []

    x, y = get_batch(train_data, cfg.batch_size, cfg.seq_len, generator)

    for i in range(cfg.max_steps):
        if i % cfg.eval_interval == 0:
            train_loss = estimate_loss(
                model,
                train_data,
                cfg.batch_size,
                cfg.seq_len,
                cfg.eval_batches,
                generator=generator,
            )
            val_loss = estimate_loss(
                model,
                val_data,
                cfg.batch_size,
                cfg.seq_len,
                cfg.eval_batches,
                generator=generator,
            )

            record = {"step": i, "train": train_loss, "val": val_loss}

            print(f"step {i}: train {train_loss:.3f}, val {val_loss:.3f}")

            history.append(record)

        if i != cfg.max_steps - 1:
            train_step(model, optimizer, x, y)

    train_loss = estimate_loss(
        model,
        train_data,
        cfg.batch_size,
        cfg.seq_len,
        cfg.eval_batches,
        generator=generator,
    )
    val_loss = estimate_loss(
        model,
        val_data,
        cfg.batch_size,
        cfg.seq_len,
        cfg.eval_batches,
        generator=generator,
    )

    record = {"step": cfg.max_steps, "train": train_loss, "val": val_loss}

    print(f"step {cfg.max_steps}: train {train_loss:.3f}, val {val_loss:.3f}")

    history.append(record)
    return history


# ===============================================================
# 下面不是练习：数据下载、字符级 tokenizer，和把一切串起来的 main()
# ===============================================================
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class CharTokenizer:
    """最简单的 tokenizer：每个字符就是一个 token。Tiny Shakespeare 只有 65 种字符，词表很小。"""

    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)


def load_shakespeare() -> str:
    path = DATA_DIR / "tinyshakespeare.txt"
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        print(f"downloading {SHAKESPEARE_URL}")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    return path.read_text()


def main() -> None:
    torch.manual_seed(0)
    text = load_shakespeare()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text))
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(
        f"{len(text):,} 个字符，词表 {tok.vocab_size}，train {n:,} / val {len(data) - n:,} tokens"
    )

    cfg = TrainConfig()
    model = GPT(
        GPTConfig(
            vocab_size=tok.vocab_size,
            max_seq_len=cfg.seq_len,
            d_model=128,
            n_heads=4,
            n_layers=4,
        )
    )
    print(
        f"模型参数: {sum(p.numel() for p in model.parameters()):,}，瞎猜时的 loss = ln({tok.vocab_size}) = "
        f"{torch.log(torch.tensor(tok.vocab_size)).item():.3f}"
    )

    prompt = torch.tensor([tok.encode("ROMEO:")])
    print("--- 训练前:")
    print(tok.decode(generate(model, prompt, max_new_tokens=200)[0]))

    start = time.time()
    train(model, train_data, val_data, cfg)
    print(f"训练用时 {time.time() - start:.0f}s")

    print("--- 训练后:")
    print(tok.decode(generate(model, prompt, max_new_tokens=500, temperature=0.8)[0]))

    DATA_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), DATA_DIR / "shakespeare_char.pt")


if __name__ == "__main__":
    main()
