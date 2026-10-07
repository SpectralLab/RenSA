# ReNVSA 实验与对照 / ReNVSA experiments and baselines

所有命令均从仓库根目录运行，输入数据放在本地 `data/`。代码包保留实验脚本和参数设置；检查点、数值结果及日志由运行脚本在本地生成，不随代码发布。 / Run all commands from the repository root and put input files in local `data/`. The package preserves the experiment scripts and parameter settings; checkpoints, numerical results, and logs are generated locally and are excluded from this code release.

## 实验流程 / Experimental pipeline

```text
外层数据划分 / Outer split
→ 仅用实测样本拟合预处理 / Fit preprocessing on measured calibration samples only
→ 全光谱 ReNVSA 及一致性筛选 / Full-spectrum ReNVSA and consistency screening
→ 仅用实测样本拟合 CARS / Fit CARS on measured calibration samples only
→ 对实测、生成和保留集使用同一 CARS 特征掩码 / Apply one fixed CARS mask to measured, synthetic, and holdout spectra
→ 用实测及生成样本拟合 RBF-SVR / Fit RBF-SVR on measured and synthetic samples
→ 预测实测保留集 / Predict the measured holdout set
```

## 主实验 / Main experiment

先运行 `coal_Q` 与确定性 SPXY 划分的快速检查。 / First run the smoke test on `coal_Q` with a deterministic SPXY split:

```powershell
python renvsa_experiments/run_renvsa.py --smoke-test
```

检查通过后运行全部 35 个外层实验；`--resume` 可跳过已完成的划分。 / After the smoke test, run all 35 outer experiments; `--resume` skips completed splits:

```powershell
python renvsa_experiments/run_renvsa.py --resume
```

也可以只运行指定数据集和划分。 / You can also run a selected subset:

```powershell
python renvsa_experiments/run_renvsa.py --tasks coal_Q soil_SOC --split-methods mc --mc-seeds 0 1 --resume
```

主实验输出位于 `renvsa_experiments/results/`，包括 `split_results.csv`、`predictions.csv`、`y_distribution_statistics.csv`、`response_sparsity.csv` 和 `summary.csv`；检查点位于其 `checkpoints/` 子目录，日志写入 `renvsa_experiments/logs/run.log`。 / Main outputs are under `renvsa_experiments/results/`, including `split_results.csv`, `predictions.csv`, `y_distribution_statistics.csv`, `response_sparsity.csv`, and `summary.csv`. Split checkpoints are in its `checkpoints/` subdirectory, and logs append to `renvsa_experiments/logs/run.log`.

## 匹配对照 / Matched controls

Bjerrum 增强对照使用与主实验相同的外层划分及仅由实测样本选出的 CARS 特征。先运行 ReNVSA；Bjerrum 脚本读取主实验 `results/split_results.csv`，并写入 `results_bjerrum/`。 / The Bjerrum augmentation baseline uses the main experiment's outer splits and measured-only CARS selections. Run ReNVSA first; the Bjerrum script reads the main `results/split_results.csv` and writes to `results_bjerrum/`.

```powershell
python renvsa_experiments/run_bjerrum_baseline.py --resume
```

随机配对对照仅改变配对规则，保留其余全光谱流程。更早的保留集与增强对照脚本见 `examples/`。 / The random-pair control changes the pairing rule while retaining the other full-spectrum steps. Earlier holdout and augmentation controls are in `examples/`.

```powershell
python renvsa_experiments/run_random_pair_control.py --resume
```

## SMOGN/SMOTER 基线 / SMOGN/SMOTER baselines

先安装 `pyproject.toml` 中固定版本的依赖，并运行连续光谱实现的自检。 / Install the dependency versions pinned in `pyproject.toml`, then self-test the continuous-spectra implementation:

```powershell
python -m pip install -e .
python renvsa_experiments/run_smogn_smoter_baselines.py --self-test
```

五个任务沿用主实验的外层划分和内层交叉验证预算。运行主实验与基线后再汇总配对指标。 / The five tasks use the main experiment's outer splits and inner cross-validation budget. Summarize paired metrics after the main and baseline runs:

```powershell
python renvsa_experiments/run_smogn_smoter_baselines.py --resume
python renvsa_experiments/summarize_smogn_smoter_comparison.py
```

每种方法使用六个预设增强配置与六种预处理方式，共 36 个候选流程；每个候选使用仅含实测样本的五折验证及 36 点 RBF-SVR 网格。结果、候选分数、依赖包哈希和实验协议写入本地 `renvsa_experiments/results_standard_smogn_smoter/`。 / Each method crosses six predeclared augmentation configurations with six preprocessing choices, yielding 36 pipeline candidates. Each candidate uses five-fold measured-only validation and the same 36-point RBF-SVR grid. Results, candidate scores, package hashes, and protocol details are written locally to `renvsa_experiments/results_standard_smogn_smoter/`.

`smogn_smoter_numeric.py` 按已发表的 SMOTER/SMOGN 算法及 UBL 0.0.9 采样规则处理数值光谱；邻居距离不使用目标值，并完整初始化生成样本。运行时生成的 manifest 记录本地和上游源码哈希及实现符合性说明。 / `smogn_smoter_numeric.py` follows the published SMOTER/SMOGN algorithms and UBL 0.0.9 sampling rules for numeric spectra. It excludes the target from neighbor distances and fully initializes generated samples. The generated manifest records local and upstream source hashes and implementation-conformance decisions.
