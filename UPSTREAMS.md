# 上游模板版本记录

以下是导入本 monorepo 时固定的上游 `main` 提交。后续上游更新不自动覆盖本地实现；
如需同步，必须新建迁移记录并重新运行对应项目的检查。

| 项目 | 仓库 | 固定提交 |
|---|---|---|
| 项目 1 | `Qi-StarterTrain/diffusion-project1-ddpm` | `56bd97e27655e1e22db5610877c8bf9e024cff96` |
| 项目 2 | `Qi-StarterTrain/diffusion-project2-samplers` | `774d640f3915c3396e034068beff318fbf421719` |
| 项目 3 | `Qi-StarterTrain/diffusion-project3-stable-diffusion` | `e07f756042ddd7329292624ad12a73b582867c18` |
| 项目 4 | `Qi-StarterTrain/diffusion-project4-flow-matching` | `0e0074f41c168d0f69225e7f1e06ed42e75e911d` |
| 项目 5 | `Qi-StarterTrain/diffusion-project5-vla-action-diffusion` | `381d52b1e826ad3adf447e864c938025cfb55c87` |

导入方式为普通文件快照，不保留上游 `.git`。本地修改应在各项目自己的提交中完成，
并在项目 README 或 `docs/migration_log.md` 中注明与上游的差异。
