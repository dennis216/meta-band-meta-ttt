# 复现与运行

## 环境

建议 Python 3.11、Linux 或 WSL2。训练脚本包含 `fcntl` 和 CUDA 调用，不承诺原生 Windows 训练兼容。发布时测试使用当前工作环境 PyTorch `2.14.0+cu130`；CHB 迁移快照记录的是 `2.11.0+cu128`。环境差异可能影响二阶梯度、数值误差与阈值附近事件，不能视为已经通过跨版本数值等价检验。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
# 先按 PyTorch 官方说明安装与驱动匹配的 CUDA wheel。
python -m pip install -e '.[test]'
```

额外历史基线需要的依赖在 `.[legacy]`；不要为了主 TUSZ 路径默认安装全部历史模型包。精确环境快照见 `environment/tested-versions.txt`，它是审计记录，不是跨平台安装锁文件。

## 路径迁移

现代 TUSZ 入口多数从文件位置确定项目根目录，旧 CHB 队列仍保留原始绝对路径。以下工具会创建一个**全新运行副本**并替换已知路径，不覆盖原仓库或已有运行目录：

```bash
python tools/prepare_runtime.py \
  --destination /path/to/meta-ttt-runtime \
  --tusz-root /data/TUSZ_v2.0.6 \
  --chb-root /data/chbmit-1.0.0 \
  --chb-cache /data/chb-cbramod-cache
cd /path/to/meta-ttt-runtime
python -m venv .venv
source .venv/bin/activate
# 安装 CUDA PyTorch 后：
python -m pip install -e '.[test]'
```

转换报告写入运行副本的 `runtime-paths.json`。归档目录仅供查阅，不作为运行目录。工具只迁移路径，不创建患者清单、不迁移模型、不改变实验算法。

迁移会保留 `src/bfa/data` 和 `tests/data` 等代码目录，只排除项目根目录的数据/输出目录、生成缓存、模型文件和符号链接。完成迁移后先运行 `python tools/verify.py`；所需 EEG 和权重按下文单独准备。

## 外部资产

1. CBraMod 预训练权重放在 `third_party/CBraMod/pretrained_weights/pretrained_weights.pth`；CHB 入口使用 `external/NeuroTTT_CBraMod/pretrained_weights/pretrained_weights.pth`。
2. 从原项目安全迁移已验证的 S1 和 SSL 暖启动 checkpoint，保持对应 `outputs/reports/tusz_meta_ttt_v1`、`tusz_meta_ttt_v2` 下的相对目录。只加载可信 checkpoint；原训练入口使用 PyTorch 完整 checkpoint 恢复。
3. 迁移/重建 `records.json`、`development_split.json`、信号缓存及 sidecar；它们不能由本代码包凭空恢复。要精确复现实验，必须使用原始划分和相同内容校验值。
4. CHB 需 `manifests/windows.parquet`、`recordings.parquet`、`seizures.parquet`、`groupkfold_cv_v1/fold_*.json` 和对应信号缓存。入口 `03_build_manifests.py` / `03b_build_windows.py` 可生成基础清单；原五折划分需单独保留。

上游权重位置（来自原迁移说明）：
https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth

原迁移说明记录的 SHA-256 为
`0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`。
下载后应核对；发布过程中没有重新下载权重。

## TUSZ 路径

准备新数据清单和缓存时：

```bash
python scripts/301_prepare_tusz_meta_ttt_v1.py audit --content-hash
python scripts/301_prepare_tusz_meta_ttt_v1.py cache --partition train
python scripts/309_materialize_tusz_signal_sidecars_v1.py --help
```

已有实验优先复用经过核对的 S1 / 缓存，不因代码迁移重新训练 S1。后续入口先用 `--help` 核对输入：

```bash
python scripts/321_train_tusz_ssl_v2.py --help
python scripts/322_calibrate_tusz_inner_v2.py --help
python scripts/401_rescore_tusz_meta_ttt_v4.py --help
python scripts/410_calibrate_tusz_meta_ttt_v4.py --help
python scripts/411_train_tusz_meta_ttt_v4.py --help
```

v4 示例（资产到位、损失尺度标定完成后）：

```bash
python scripts/411_train_tusz_meta_ttt_v4.py \
  --objective band --conditions a b c \
  --loss-scale /path/to/band-loss-scale.json \
  --reference-probabilities /path/to/s1-reference.parquet \
  --patients-per-batch 1 --epochs 2 --seed 3407 \
  --output outputs/reports/tusz_meta_ttt_v4/runs/development/band_f
```

Mask 改为 `--objective mask` 并使用其独立标定文件。上述路径是示例，须使用校准入口实际产出的格式和文件。`configs/tusz_meta_ttt_v4/unit_loss_scale.json` 是单位系数文件，不能冒充标定结果。

评价/统计入口为 `323`、`401`、`412`、`413`、`414`、`415`、`416`。shell 队列预设本地目录、输出名称和日志路径；运行前逐项检查。不要执行 `scripts/*.sh` 通配批量启动。benchmark 和队列脚本也会启动耗时作业。

## CHB-MIT 路径

```bash
export BFA_ROOT="$PWD"
export BFA_CACHE_ROOT=/data/chb-cbramod-cache
export NEUROTTT_CODE_ROOT="$PWD/external/NeuroTTT_CBraMod"
python external/NeuroTTT_CBraMod/chbmit_groupkfold_meta_train.py --help
python external/NeuroTTT_CBraMod/chbmit_groupkfold_meta_evaluate.py --help
python scripts/280_retrain_band_ttt_v2.py --help
python scripts/281_evaluate_retrained_band_ttt_v2.py --help
```

早期 `230` 系列 TU 转移实验引用历史生成清单，保留用于追溯；它不等同于 TUSZ v1–v4 的完整官方分区协议。共享模块还保留少量历史模型 adapter，相关权重/第三方实现未捆绑，不能把它们视为主 Meta-TTT 必需资产。

## 验证范围

见 [VALIDATION.md](VALIDATION.md)。发布测试验证代码和无数据数值用例，不代替完整 GPU 训练、跨 seed 重训、事件置信区间或论文结果复核。
