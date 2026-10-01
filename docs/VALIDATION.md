# Publication validation

本记录针对代码整理，不代表重新完成训练或验证论文性能。

发布检查使用 Python 3.11 / Linux（WSL2），当前安装版本见 `environment/tested-versions.txt`。

| 检查 | 结果 |
|---|---|
| TUSZ 全部现有测试、data、evaluation、preprocessing、contracts | 134 passed，1 warning |
| 新增路径迁移测试（保留原件、拒绝覆盖） | 1 passed |
| Python AST 语法解析 | 350 文件通过 |
| 原始数据、权重和常见凭据模式扫描 | 未发现禁止发布项 |
| Router / idea3 实验入口 | 已排除 |

执行的核心命令：

```bash
PYTHONPATH=src python -m pytest tests/tusz_meta_ttt tests/data tests/evaluation tests/preprocessing tests/test_contracts.py -q
PYTHONPATH=src python -m pytest tests/test_publication_paths.py -q
python tools/check_release.py
```

检查不包含下载 EEG、训练权重、完整训练、GPU 并行一致性重测或所有历史模型测试。部分 `tests/models` 需要未发布的 checkpoint 或其他模型依赖，不能把上述 135 项理解为整个测试目录全部通过。

源文件尽量保持原研究版本。`docs/source-manifest.json` 保存复制前 SHA-256，包装新增 README、工具、依赖声明和测试单独维护。旧入口的本机路径保留在源代码中，通过运行副本工具迁移；共享模块中的历史模型适配器未被改写为新方法。

CHB 外部实现来自 2026-09-05 冻结包，TUSZ 来自当前研究工作目录。归档脚本与现代入口可能使用不同指标语义，结果不可直接拼接。完整复现仍需原始患者划分、缓存版本与 checkpoint；它们未公开上传。
