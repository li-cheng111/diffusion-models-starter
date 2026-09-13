# Monorepo 迁移记录

## 2026-09-13

- 建立 `projects/`、`shared/`、`requirements/`、`docs/` 和 `scripts/`。
- 将 Project 1 迁入 `projects/project1_ddpm/`。
- 将 Project 2 迁入 `projects/project2_samplers/`。
- 按固定上游提交导入 Project 3、4、5，不带入嵌套 `.git`。
- 将 Project 1/2 的导入入口改为可从仓库根目录使用 `python -m` 运行。
