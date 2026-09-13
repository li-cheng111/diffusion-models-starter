"""观测 → 模型输入的统一约定。

train.py 和 eval.py 都从这里取，避免训练和评估用了不同的条件表示——
那种 bug 表现为"训练 loss 很漂亮但闭环成功率接近 0"，极难查。

两种条件模式：

| use_vision | 送进模型的 state | 维度 | 干扰物信息 |
|------------|-----------------|------|-----------|
| False（阶段 2） | concat(agent_pos, target_pos) | 4 | 无 → 学不会避障 |
| True （阶段 3） | agent_pos（本体感知）        | 2 | 从 image 来 |

阶段 3 之所以只喂 agent_pos，是因为这才是真实 VLA 的设置：
机器人知道自己的关节角，但目标和障碍都得从相机里看出来。
"""

import numpy as np
import torch


def state_dim_for(use_vision: bool) -> int:
    """模型 state_dim 应该配多少。"""
    return 2 if use_vision else 4


def state_from_obs(obs: dict, use_vision: bool) -> np.ndarray:
    """env.observation() 的输出 → (state_dim,) 的 numpy 向量（给 eval 用）。"""
    if use_vision:
        return np.asarray(obs["state"], dtype=np.float32)
    return np.concatenate([obs["state"], obs["goal"]]).astype(np.float32)


def state_from_batch(batch: dict, use_vision: bool) -> torch.Tensor:
    """DataLoader 的 batch → (B, state_dim) tensor（给 train 用）。"""
    if use_vision:
        return batch["state"]
    return torch.cat([batch["state"], batch["goal"]], dim=-1)
