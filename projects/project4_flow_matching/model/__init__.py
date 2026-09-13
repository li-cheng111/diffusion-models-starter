"""Project 4 的 backbone：conditional UNet 与简化 DiT。

两者接口一致：`model(x, t, y) -> v`，其中
    x: (B, 3, 32, 32)
    t: (B,) 连续时间，取值 [0, 1]
    y: (B,) 类别标签，0..num_classes-1 是真实类别，num_classes 是 CFG 的 null token
"""

from .dit import SimpleDiT
from .unet import ConditionalUNet

__all__ = ['ConditionalUNet', 'SimpleDiT']
