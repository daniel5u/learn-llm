Things to learn:

- [ ]真正的问题是"训练不足"：一个 token 在训练数据里几乎不出现，它的 embedding 就几乎没被更新过，基本停留在初始化附近的状态。如果它某天出现在输入里，模型的行为会很奇怪。我们最开始提到的 glitch token（最有名的是 GPT-2 词表里的 " SolidGoldMagikarp"）就是这么来的。2024 年有一篇论文 Fishing for Magikarp 专门研究怎么自动找出这类训练不足的 token。
  训练 tokenizer 时就清理。 比如 PickyBPE（2024）：每次合并出新 token 后，检查它的组成部分是否几乎只出现在这个新 token 里；是的话，就把那个中间 token 从词表里删掉。
- Something
