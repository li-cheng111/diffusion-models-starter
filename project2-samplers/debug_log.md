# 项目 2 调试日志

## 1. 调度策略 buffer 命名不一致

- 现象：项目 2 starter 期望单数 buffer 名称，而项目 1 使用
  `sqrt_alphas_cumprod` 和 `sqrt_one_minus_alphas_cumprod`。
- 验证：兼容性检查确认复数名称对应的系数有效。
- 修复：增加统一的别名访问器，不修改项目 1 代码。
- 经验：采样器应依赖系数含义，而不是变量拼写。

## 2. 嵌套 EMA checkpoint 格式

- 现象：starter 直接尝试加载 `checkpoint["ema"]`。
- 验证：项目 1 保存的是 `ema = {decay, model}`。
- 修复：增加同时支持嵌套和扁平 EMA state dictionary 的加载器。
- 经验：开始长时间推理前必须先验证 checkpoint 结构。

## 3. FID 不公平和 NFE 计数不准确

- 现象：starter 使用了增强后的真实图，不同配置之间 RNG 状态不同，可能超过
  5,000 张真实图，并将 DPM 终止步报告为两次模型调用。
- 修复：固定真实图子集、每个配置重置 RNG、强制 real/generated 数量相等，
  并将 DPM-Solver-2 报告为 `2S-1`。
- 经验：Pareto 对比必须控制数据和计算量统计。

## 4. DPM 轨迹没有共享初始噪声

- 现象：starter 创建了公共噪声，但 DPM 的 `sample()` 又将其替换。
- 修复：为所有 sampler 增加 `initial_x` 参数，所有轨迹运行都通过该接口传入。
- 经验：当 RNG 消耗方式不同时，必须显式传入受控 tensor。
