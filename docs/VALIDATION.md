# Publication validation

本记录针对代码整理，不代表重新完成训练或验证论文性能。

发布检查使用 Python 3.11 / Linux（WSL2），当前安装版本见 `environment/tested-versions.txt`。

| 检查 | 结果 |
|---|---|
| TUSZ 全部现有测试、data、evaluation、preprocessing、contracts | 134 passed，1 warning |
| 发布与迁移回归测试 | 4 passed |
| Python AST 语法解析 | 352 文件通过 |
| 干净 Git 索引导出副本运行测试 | 138 passed，1 warning |
| 完整运行副本迁移 | 62 文件路径已替换，数据处理包保留 |
| 迁移后入口 smoke checks | TUSZ 数据准备、v4 训练、CHB Meta 训练的 `--help` 均通过 |
| 原始数据、权重和常见凭据模式扫描 | 未发现禁止发布项 |
| Router / idea3 实验入口 | 已排除 |

执行的核心命令：

```bash
python tools/verify.py
```

检查不包含下载 EEG、训练权重、完整训练、GPU 并行一致性重测或所有历史模型测试。部分 `tests/models` 需要未发布的 checkpoint 或其他模型依赖，不能把上述 138 项理解为整个测试目录全部通过。

2026-10-02 修正：初始 `.gitignore` 的 `data/` 规则误排除了 `src/bfa/data` 和 `tests/data`，迁移工具也使用了同样过宽的目录过滤。现改为仅排除根目录数据，补齐 Git 中缺失的 8 个源码/测试文件。验证使用 `git checkout-index` 导出的干净目录，不依赖工作树未提交文件；发布检查还会核对源码清单中的文件是否存在。初版 135 项测试是在本地完整副本运行，不能证明初版远端仓库完整。

保留的一条 warning 来自原有截断梯度测试将 requires-grad tensor 转为标量的断言计算；该测试通过。本轮没有为了消除 warning 改动研究算法。

源文件尽量保持原研究版本。`docs/source-manifest.json` 保存复制前 SHA-256，包装新增 README、工具、依赖声明和测试单独维护。旧入口的本机路径保留在源代码中，通过运行副本工具迁移；共享模块中的历史模型适配器未被改写为新方法。

CHB 外部实现来自 2026-09-05 冻结包，TUSZ 来自当前研究工作目录。归档脚本与现代入口可能使用不同指标语义，结果不可直接拼接。完整复现仍需原始患者划分、缓存版本与 checkpoint；它们未公开上传。
