# LDM §3–4 与 SDXL 前半部分阅读笔记

## 论文与范围

- **High-Resolution Image Synthesis with Latent Diffusion Models**，Rombach et al.，CVPR 2022。
- **SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis**，Podell et al.，2023。
- 本笔记覆盖 LDM 的 latent diffusion 设计与 cross-attention 条件化，以及 SDXL 的架构改动前半部分。

## LDM §3–4：从像素空间到 latent 空间

像素空间扩散的每个 denoising step 都在高维 RGB 网格上运行，计算昂贵且冗余。LDM 先训练一个感知压缩 autoencoder：encoder `E` 把图像 `x` 映射到较小的 latent `z=E(x)`，decoder `D` 重构 `x≈D(z)`；扩散模型只在 `z` 上学习。这样既保留感知上重要的语义结构，也显著降低 UNet 的空间计算量。

前向过程为 `q(z_t|z_{t-1})=N(sqrt(1-β_t)z_{t-1}, β_t I)`，常用闭式写法
`z_t=√ᾱ_t z_0 + √(1-ᾱ_t) ε`。UNet 学习噪声回归目标

`L_simple = E_{z_0,ε,t,c}[ || ε - ε_θ(z_t,t,c) ||² ]`。

其中 `c` 可为空，也可由文本 encoder 得到。文本条件通过 cross-attention 注入：
`Q=W_Q h` 来自当前 latent feature，`K=W_K c`、`V=W_V c` 来自文本 token embedding，
`softmax(QKᵀ/√d)V` 将 token 语义写回空间位置。classifier-free guidance 用 unconditional
和 conditional 两个预测做线性外推，提高 prompt adherence。

## LDM 的关键贡献与限制

- **贡献**：在低维 latent 上扩散，计算/显存更可控；cross-attention 使文本、布局、类别等条件共享同一 UNet 接口。
- **限制**：autoencoder 的压缩误差不可由 diffusion 完全恢复；latent scaling、VAE 解码质量和文本 tokenizer 都会影响最终细节。
- **实践启示**：Project 3 手写推理必须显式除以 SD 1.5 的 `0.18215`，否则 VAE decoder 的输入尺度错误。

## SDXL 前半部分

SDXL 不是简单增大 SD 1.5 的 UNet，而是同时扩大 base UNet、改进 conditioning，并加入更高分辨率的两阶段流程。文本侧使用两个 CLIP text encoder，把不同 tokenizer/embedding 空间的信息拼接，并加入 pooled text embedding 作为额外的全局条件；time embedding 还接收原始尺寸、裁剪坐标和目标尺寸，使模型能区分相同 latent 在不同构图条件下的含义。

SDXL 的 base model 负责生成 latent，refiner model 只在较低噪声阶段继续去噪，以较小的计算代价恢复高频纹理。该设计把“全局构图”和“局部细节”分工，适合高分辨率，但需要更多显存、两个模型和更严格的尺寸元数据。

## 与 Project 3 的连接

1. A 部分的 `(1,4,64,64)` latent 是 LDM 压缩后的扩散空间，而不是 RGB 像素。
2. D 的 LoRA 只修改 UNet attention projection，保留 VAE 和 text encoder，等价于在固定 latent/文本接口上学习低秩风格偏移。
3. E 的 attention challenge 直接观察 `QKᵀ` 对 token 的空间响应，可用来分析 CFG、层级和 timestep 对语义绑定的影响。
4. SDXL 的 pooled embedding 与尺寸条件提示了下一步扩展：若迁移到 SDXL，不能只替换 UNet 权重，必须同步处理两个 text encoder 和 added conditioning。

## 局限与可复现实验问题

LDM/SDXL 的论文指标依赖大规模数据、特定 VAE 和采样器；课程实验的单 prompt 图像不能等价为 FID 结论。因而本项目把 model revision、seed、scheduler、prompt、运行环境和产物 SHA256 写入 metadata，并将 AutoDL full run 与本机 smoke 分开记录。
