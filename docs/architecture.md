# 仓库架构说明

五个 Project 是可独立运行的边界单元。`shared/` 只提供不包含具体模型结构的基础设施；
项目之间的复用通过明确的 Python 包导入或标准化结果文件完成，不通过隐式的相对路径
猜测完成。

从仓库根目录运行项目命令，路径由 `shared.paths` 或项目配置显式解析。Project 2 的
采样器可以读取 Project 1 的模型和 schedule，但 sampler 代码仍归 Project 2 所有。
