# Project 5 debug log

这些记录来自本次 AutoDL/Windows 实际运行，不是预设示例。

## 1. Windows 控制台编码错误

- **现象**：第一次运行 `check_setup.py` 时，中文勾号输出触发 `UnicodeEncodeError: 'gbk' codec can't encode character`。
- **定位**：逻辑检查已经通过，错误发生在 stdout 编码而不是模型或环境。
- **处理**：入口处按 `sys.stdout.reconfigure(encoding="utf-8")`，同时保留普通终端的 fallback。
- **教训**：自检脚本也要考虑不同终端编码，否则“通过”会被输出层错误掩盖。

## 2. 测试从仓库根目录导入失败

- **现象**：`pytest -q tests` 报 `ModuleNotFoundError: eval`，因为测试文件位于项目子目录而源文件不是已安装包。
- **定位**：测试运行目录是仓库根，Python path 中没有 Project 5 目录。
- **处理**：测试入口将项目根加入 `sys.path`，再以 `from eval import evaluate` 等方式调用；AutoDL 上最终 `4 passed`。
- **教训**：仓库作业应同时支持“项目目录运行”和“仓库根目录运行”。

## 3. AutoDL 完整 Git clone 中断

- **现象**：在 AutoDL `/root/autodl-tmp` 克隆大仓库时，传输到约 52% 报 GitHub HTTP/2 RPC framing/early EOF。
- **定位**：不是代码错误；实例数据盘只剩约 4.7GB，完整历史也没有必要。
- **处理**：本地先 push 分支，再只上传 Project 5 的源代码、配置和测试到独立 run tree；保留远程 tmux 训练日志，不上传 600MB demo 数据回本地。
- **教训**：云端实验应优先使用浅克隆/最小运行树，并持续观察磁盘空间。

## 4. Global pooling 视觉成功率偏低

- **现象**：第一版三层 CNN + `AdaptiveAvgPool2d(1)` loss 已收敛到约 0.118，但 100 集闭环成功率只有 54%；exec_steps=1 仍为 54%。
- **定位**：全局平均池化对平移近似不敏感，target/distractor 的二维位置被压掉。
- **处理**：改为 `AdaptiveAvgPool2d(4)` 保留 4×4 网格，重新训练 10,000 steps；最终 loss 约 0.097，成功率升到 71%。
- **教训**：闭环任务不能只看训练 loss；视觉任务的空间信息应显式保留。

## 5. 过长 action chunk 退化

- **现象**：H=32 训练看似收敛（loss 约 0.123），但闭环成功率仅 14%，85% episode 超时。
- **定位**：成功 expert 轨迹中位数 26 步，H=32 的重复末动作 padding 已达 51.7%。
- **处理**：采用 H=16 作为最终配置（padding 27.5%），并在报告中保留 H=32 失败消融。
- **教训**：chunk 长度应与任务时间尺度匹配，不能用更长 horizon 代替更好的表征。
