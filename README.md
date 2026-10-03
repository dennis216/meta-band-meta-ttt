# Meta-Band / Meta-TTT

Research code for seizure detection and self-supervised test-time adaptation on **CHB-MIT** and **TUSZ**, using CBraMod representations.

本仓库整理 Meta-Band / Meta-TTT 研究线的训练、评价、机制分析和性能测试代码。保留 CHB-MIT 历史实现以及 TUSZ v1–v4，不包含 Router / idea3 研究代码。各版本是独立实验条件，不能混合其数据划分、checkpoint、阈值或时间语义。

## 从哪里开始

| 内容 | 位置 |
|---|---|
| 安装、数据与运行步骤 | [docs/REPRODUCING.md](docs/REPRODUCING.md) |
| 全部入口导航 | [docs/SCRIPT_INDEX.md](docs/SCRIPT_INDEX.md) |
| 发布检查与已知边界 | [docs/VALIDATION.md](docs/VALIDATION.md) |
| 源码来源与原始 SHA-256 | [docs/source-manifest.json](docs/source-manifest.json) |
| 第三方授权说明 | [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) |

```text
src/bfa/                         公共数据、预处理、模型、评分与训练模块
src/bfa/tusz_meta_ttt/            TUSZ v1 及共享组件
src/bfa/tusz_meta_ttt_v2/         联合 Meta、F/C 模式、批处理与并行优化
src/bfa/tusz_meta_ttt_v3/         对齐与损害约束
src/bfa/tusz_meta_ttt_v4/         配对收益、Frozen 保持与路径分解
scripts/                        保留编号的训练、评价、统计与 benchmark 入口
external/NeuroTTT_CBraMod/       CHB-MIT GroupKFold / Meta-Band 实现
third_party/CBraMod/             TUSZ 使用的 CBraMod 模型依赖
archive/chb-20260905/            与当前入口有差异的 CHB 冻结脚本
configs/ + protocols/           已有配置与协议
tests/                          单元、数值与状态语义检验
tools/                          跨机器运行副本与发布检查工具
```

## 快速检查

推荐 Linux / WSL2、Python 3.11。先安装适合显卡驱动的 PyTorch，再执行：

```bash
python -m pip install -e '.[test]'
python tools/verify.py
```

上述测试不需要 EEG 数据或训练 checkpoint。GPU 训练与完整实验复现需要另行准备数据、CBraMod 预训练权重、S1 checkpoint、暖启动 head 和对应清单，详见复现说明。部分旧脚本依赖原机器绝对路径；`tools/prepare_runtime.py` 会生成路径替换后的独立运行副本，不修改发布源码。

`tools/verify.py` 使用当前 Python 环境执行发布检查和 CPU 测试，任一步失败即返回非零退出码，不启动训练。可在克隆后或提交前运行。

## 实验版本

| 版本 | 主要内容 |
|---|---|
| CHB-MIT | GroupKFold、Band 辅助任务、窗口/记录/患者级历史适应实验 |
| TUSZ v1 | 数据审计、监督 S0/S1、SSL、早期 Meta 和事件评价 |
| TUSZ v2 | Encoder / detector / SSL head outer 范围；F 与 C 分开训练；吞吐优化 |
| TUSZ v3 | Band / Mask 的 Post-BCE、alignment、damage 消融 |
| TUSZ v4 | 修正事件评分；同 query pre/post 配对收益；保护 Frozen 能力 |

F 表示当前 chunk 更新服务未来 chunk；C 表示获得整个当前 chunk 后适应并回顾性预测。C 不能按 F 的在线时间语义解读。v2 及后续复用信号缓存，不据此声称原始 EDF 到报警的逐样本严格因果性。

这是代码发布，不包含患者级结果或新的疗效/性能结论。历史协议描述的是实验设计，不代表每项实验均已完成或通过。整理过程没有启动训练，也没有改写原研究目录。

## 数据与权重

不上传原始 EEG、患者清单、缓存、模型权重、逐窗口概率、日志或邮件。TUSZ 需按数据提供方要求自行获得访问权。CBraMod 第三方许可证随源码保留；本项目原创代码尚未另行指定开源许可证，公开可读不等同于额外授予商业或再许可权利。
