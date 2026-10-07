# Respond Spectra / 响应邻域虚拟样本光谱扩增

本仓库提供响应邻域虚拟样本扩增（RenSA）实验的代码，用于小样本光谱建模。这里仅发布代码；输入数据、实验结果、预测值、图像、日志和论文文件均不包含在内。

This repository provides the code for response-neighbor virtual sample augmentation (ReNVSA) experiments in small-sample spectral modeling. It is a code-only release: input data, results, predictions, figures, logs, and manuscript files are excluded.

## 功能 / Features

- 光谱预处理：SNV、MSC、Savitzky–Golay 平滑及导数。 / Spectral preprocessing: SNV, MSC, and Savitzky–Golay smoothing and derivatives.
- 响应邻域配对与光谱、响应值联合插值。 / Response-neighborhood pairing and joint interpolation of spectra and responses.
- 可选的局部光谱扰动及一致性筛选。 / Optional local spectral perturbation and consistency screening.
- 交叉验证、特征选择和 PLSR 等模型评估工具。 / Cross-validation, feature selection, and model evaluation utilities including PLSR.

输入数据通常按行存储样本，按列存储波长或波数。 / Input tables usually store samples in rows and wavelengths or wavenumbers in columns.

## 目录 / Repository layout

| Directory | 中文说明 | English description |
| --- | --- | --- |
| `respond_spectra/` | 增强、预处理、特征选择与评估核心代码。 | Core augmentation, preprocessing, feature selection, and evaluation code. |
| `rensa_experiments/` | ReNSA 主实验、基线和敏感性分析。 | Main RenSA protocol, baselines, and sensitivity analyses. |
| `examples/` | 其他实验、对比、搜索和诊断脚本。 | Additional experiments, comparisons, searches, and diagnostics. |
| `tests/` | 单元测试与快速检查。 | Unit and smoke tests. |

## 文件清单 / File guide

以下列出代码包中的全部 45 个文件。`examples/` 中的早期实验不需要按文件名顺序全部运行；读取已有预测结果的诊断脚本应在相应实验之后运行。包内没有绘图或论文制表脚本；实验运行时会在本地生成数值结果 CSV。

All 45 files are listed below. The earlier experiments in `examples/` need not run in filename order. Diagnostics that read saved predictions require their source experiments to run first. The package has no plotting or manuscript-table scripts; experiments can generate numerical CSV results locally.

### 根目录与算法库 / Root and core library

| File | 中文用途 | English purpose |
| --- | --- | --- |
| [.gitignore](.gitignore) | 阻止本地数据、结果、图表、日志和缓存进入 Git。 | Excludes local data, results, figures, logs, and caches from Git. |
| [README.md](README.md) | 安装、运行说明和文件清单。 | Installation, usage, and file guide. |
| [pyproject.toml](pyproject.toml) | Python 包元数据、依赖和测试配置。 | Python package metadata, dependencies, and test configuration. |
| [respond_spectra/__init__.py](respond_spectra/__init__.py) | 导出实验脚本调用的公开接口。 | Exposes the public API used by experiment scripts. |
| [respond_spectra/augment.py](respond_spectra/augment.py) | 响应邻域虚拟光谱生成与一致性筛选。 | Response-neighborhood augmentation and consistency screening. |
| [respond_spectra/datasets.py](respond_spectra/datasets.py) | 读取光谱 CSV 数据。 | Loads spectral CSV datasets. |
| [respond_spectra/deep.py](respond_spectra/deep.py) | 一维 CNN 和小型 ResNet 回归模型。 | One-dimensional CNN and compact ResNet regressors. |
| [respond_spectra/evaluation.py](respond_spectra/evaluation.py) | 交叉验证及回归模型评估。 | Cross-validation and regression evaluation utilities. |
| [respond_spectra/feature_selection.py](respond_spectra/feature_selection.py) | CARS、相关性特征和区间选择。 | CARS, correlation-based features, and interval selection. |
| [respond_spectra/preprocessing.py](respond_spectra/preprocessing.py) | SNV、MSC 和 Savitzky–Golay 等预处理。 | SNV, MSC, Savitzky–Golay, and related preprocessing. |

### 主实验与正式对比 / Main experiments and baselines

| File | 中文用途 | English purpose |
| --- | --- | --- |
| [renvsa_experiments/README.md](renvsa_experiments/README.md) | 实验流程、命令和输出位置。 | Protocol, commands, and output locations. |
| [renvsa_experiments/run_renvsa.py](renvsa_experiments/run_renvsa.py) | 全光谱 ReNVSA 主实验，实测样本选 CARS，RBF-SVR 保留集评估。 | Main full-spectrum ReNVSA experiment with measured-only CARS and RBF-SVR holdout evaluation. |
| [renvsa_experiments/run_random_pair_control.py](renvsa_experiments/run_random_pair_control.py) | 随机配对对照，保留主实验其余流程。 | Random-pair control with the remaining main protocol preserved. |
| [renvsa_experiments/run_bjerrum_baseline.py](renvsa_experiments/run_bjerrum_baseline.py) | Bjerrum 增强对照，沿用主实验划分和 CARS 特征。 | Bjerrum augmentation baseline using the main splits and CARS features. |
| [renvsa_experiments/run_smogn_smoter_baselines.py](renvsa_experiments/run_smogn_smoter_baselines.py) | SMOGN/SMOTER 对照，匹配主实验的验证预算。 | SMOGN/SMOTER baselines with a matched validation budget. |
| [renvsa_experiments/smogn_smoter_numeric.py](renvsa_experiments/smogn_smoter_numeric.py) | 连续数值光谱的 SMOGN/SMOTER 实现。 | SMOGN/SMOTER implementation for continuous spectral features. |
| [renvsa_experiments/summarize_smogn_smoter_comparison.py](renvsa_experiments/summarize_smogn_smoter_comparison.py) | 汇总配对预测及 bootstrap 指标。 | Summarizes paired predictions and bootstrap metrics. |
| [renvsa_experiments/analyze_threshold_one_at_a_time.py](renvsa_experiments/analyze_threshold_one_at_a_time.py) | 逐个改变一致性阈值，分析校准集敏感性。 | Changes one consistency threshold at a time on calibration data. |
| [renvsa_experiments/analyze_threshold_grid.py](renvsa_experiments/analyze_threshold_grid.py) | 三个阈值的 3×3×3 校准集网格分析。 | Evaluates a 3×3×3 threshold grid on calibration data. |
| [renvsa_experiments/analyze_threshold_holdout.py](renvsa_experiments/analyze_threshold_holdout.py) | 检查 27 组阈值的 SPXY 保留集敏感性。 | Checks 27 threshold combinations on an SPXY holdout set. |

### 其他实验、搜索与诊断 / Additional experiments, searches, and diagnostics

| File | 中文用途 | English purpose |
| --- | --- | --- |
| [examples/run_coal_q_augmentation_demo.py](examples/run_coal_q_augmentation_demo.py) | 煤发热量数据的 ReNVSA 入门示例。 | Introductory ReNVSA demo using coal heating-value data. |
| [examples/holdout_bp_ks_spxy.py](examples/holdout_bp_ks_spxy.py) | KS/SPXY 划分下的 BP 保留集实验。 | BP holdout experiment with KS/SPXY splits. |
| [examples/holdout_cars_svr_augmentation_selection.py](examples/holdout_cars_svr_augmentation_selection.py) | 训练集内选择增强配置，再评估 CARS-SVR。 | Selects augmentation on training data before CARS-SVR holdout evaluation. |
| [examples/holdout_cars_svr_ks_spxy.py](examples/holdout_cars_svr_ks_spxy.py) | KS/SPXY 划分下的 CARS-SVR 增强实验。 | CARS-SVR augmentation with KS/SPXY holdout splits. |
| [examples/holdout_cars_svr_model_selection.py](examples/holdout_cars_svr_model_selection.py) | 仅用训练集选择 CARS 复杂度、模型和增强方案。 | Selects CARS complexity, model, and augmentation using training data only. |
| [examples/holdout_model_comparison_train_only.py](examples/holdout_model_comparison_train_only.py) | 统一训练集选择规则下比较回归模型。 | Compares regressors under a shared training-only selection rule. |
| [examples/holdout_plsr_svr_ks_spxy.py](examples/holdout_plsr_svr_ks_spxy.py) | KS/SPXY 保留集上的 PLSR、SVR 与增强对比。 | Compares PLSR, SVR, and augmentation on KS/SPXY holdouts. |
| [examples/holdout_unified_train_only_protocol.py](examples/holdout_unified_train_only_protocol.py) | 多数据集统一的训练集选择和保留集评估流程。 | Unified training-only selection and holdout evaluation across datasets. |
| [examples/compute_holdout_cars_svr_predictions.py](examples/compute_holdout_cars_svr_predictions.py) | 计算 CARS-SVR 保留集预测和配对 bootstrap 区间。 | Computes CARS-SVR holdout predictions and paired bootstrap intervals. |
| [examples/run_cross_split_robustness.py](examples/run_cross_split_robustness.py) | 多种外层划分下的配对稳健性实验。 | Paired robustness experiment across outer splits. |
| [examples/preprocessing_sensitivity_cv.py](examples/preprocessing_sensitivity_cv.py) | 预处理方案的交叉验证敏感性分析。 | Cross-validated preprocessing sensitivity analysis. |
| [examples/preprocessing_augmentation_effect_cv.py](examples/preprocessing_augmentation_effect_cv.py) | 预处理与增强组合的交叉验证分析。 | Cross-validated analysis of preprocessing and augmentation combinations. |
| [examples/search_bp_feature_models.py](examples/search_bp_feature_models.py) | 比较 PCA-BP、PLS-BP 和 CARS-BP 特征建模。 | Compares PCA-BP, PLS-BP, and CARS-BP feature models. |
| [examples/search_ensemble_pls_bp.py](examples/search_ensemble_pls_bp.py) | 搜索集成 PLS-BP 和残差 BP 模型。 | Searches ensemble PLS-BP and residual BP models. |
| [examples/search_response_augmentation_strategy.py](examples/search_response_augmentation_strategy.py) | 搜索响应驱动增强策略及其稳定性。 | Searches response-driven augmentation strategies and stability. |
| [examples/search_cars_svr_augmentation.py](examples/search_cars_svr_augmentation.py) | 搜索 CARS、RBF-SVR 和增强组合。 | Searches combinations of CARS, RBF-SVR, and augmentation. |
| [examples/run_augmentation_control_comparison.py](examples/run_augmentation_control_comparison.py) | 计算配对、噪声和邻域增强对照指标。 | Computes pairing, noise, and neighborhood control metrics. |
| [examples/run_coal_q_mechanism_controls.py](examples/run_coal_q_mechanism_controls.py) | 计算煤发热量机制消融和固定对照指标。 | Computes coal heating-value mechanism ablations and fixed controls. |
| [examples/compute_endpoint_neighborhood_diagnostics.py](examples/compute_endpoint_neighborhood_diagnostics.py) | 计算不同预测目标的响应邻域诊断。 | Computes response-neighborhood diagnostics across endpoints. |
| [examples/run_model_sensitivity_cv.py](examples/run_model_sensitivity_cv.py) | 比较模型家族的交叉验证表现。 | Compares model families by cross-validation. |
| [examples/run_modeling_pipeline_sensitivity_cv.py](examples/run_modeling_pipeline_sensitivity_cv.py) | 比较预处理、特征选择和模型组合。 | Compares preprocessing, feature selection, and model combinations. |
| [examples/audit_cross_split_sparsity.py](examples/audit_cross_split_sparsity.py) | 利用已有划分与预测复核响应稀疏度，不重新拟合。 | Audits response sparsity from saved splits and predictions without refitting. |
| [examples/compute_existing_holdout_error_diagnostics.py](examples/compute_existing_holdout_error_diagnostics.py) | 从已有预测计算偏差、斜率和 SEP 等指标。 | Computes bias, slope, SEP, and related metrics from saved predictions. |

### 测试 / Tests

| File | 中文用途 | English purpose |
| --- | --- | --- |
| [tests/test_augmentation.py](tests/test_augmentation.py) | 检查核心增强、预处理、数据读取和模型接口。 | Tests augmentation, preprocessing, data loading, and model interfaces. |
| [tests/test_standard_smogn_smoter.py](tests/test_standard_smogn_smoter.py) | 检查 SMOGN/SMOTER 的候选预算与算法行为。 | Tests SMOGN/SMOTER candidate budgets and algorithm behavior. |

## 快速开始 / Quick start

在仓库根目录安装项目。 / Install the project from the repository root:

```bash
python -m pip install -e .
```

将输入数据放入本地 `data/`。入门示例需要 `data/coal_Q.csv`；每行一个样本，光谱列名为 `x1` 至 `xN`，响应列名为 `y`。 / Put input files in local `data/`. The demo requires `data/coal_Q.csv`, with one sample per row, spectral columns `x1` through `xN`, and response column `y`.

```bash
python examples/run_coal_q_augmentation_demo.py
```

也可用 `--data` 指定同结构文件；其他实验使用各自的数据读取方式。 / Use `--data` for another file with the same layout; other experiments use their own dataset loaders:

```bash
python examples/run_coal_q_augmentation_demo.py --data data/coal_Q.csv
```

运行测试。 / Run tests:

```bash
python -m pytest
```

可选的 PyTorch 快速测试默认跳过。PowerShell 中手动启用： / The optional PyTorch smoke test is skipped by default. Enable it in PowerShell:

```powershell
$env:RESPOND_SPECTRA_RUN_TORCH_TESTS = '1'
python -m pytest tests/test_augmentation.py::test_shallow_1d_cnn_regressor_smoke
```

在 Bash 中启用： / Enable it in Bash:

```bash
RESPOND_SPECTRA_RUN_TORCH_TESTS=1 python -m pytest tests/test_augmentation.py::test_shallow_1d_cnn_regressor_smoke
```

## 复现实验 / Reproduce the experiments

安装项目并将所需文件放入 `data/` 后，从仓库根目录运行。脚本保留原有划分、随机种子、预处理、增强、CARS 和模型选择设置。 / Install the project, place the required inputs in `data/`, and run from the repository root. The original split, seed, preprocessing, augmentation, CARS, and model-selection settings are preserved.

| 中文实验 | Experiment | Script |
| --- | --- | --- |
| 全光谱 RenSA | Full-spectrum RenSA | `renvsa_experiments/run_rensa.py` |
| Bjerrum 对照 | Matched Bjerrum baseline | `renvsa_experiments/run_bjerrum_baseline.py` |
| SMOTER/SMOGN 对照 | Standard SMOTER/SMOGN baselines | `renvsa_experiments/run_smogn_smoter_baselines.py` |
| 随机配对对照 | Matched random-pair control | `renvsa_experiments/run_random_pair_control.py` |
| 早期保留集与增强对照 | Earlier holdout and augmentation controls | `examples/` |
| 一致性阈值敏感性 | Threshold sensitivity | `renvsa_experiments/analyze_threshold_*.py` |

主实验和几个对照的常用命令如下；详细顺序及输出位置见 [实验说明](renvsa_experiments/README.md)。 / Common commands for the main experiment and baselines are below; see the [experiment guide](renvsa_experiments/README.md) for run order and output locations.

```bash
python rensa_experiments/run_renvsa.py --resume
python rensa_experiments/run_bjerrum_baseline.py --resume
python rensa_experiments/run_random_pair_control.py --resume
python rensa_experiments/run_smogn_smoter_baselines.py --self-test
python rensa_experiments/run_smogn_smoter_baselines.py --resume
```

SMOTER/SMOGN 依赖版本固定在 `pyproject.toml`。主实验与基线运行后，`summarize_smogn_smoter_comparison.py` 可计算配对对比指标。所有生成的结果目录均被 Git 忽略。 / SMOTER/SMOGN dependency versions are pinned in `pyproject.toml`. After the main and baseline runs, `summarize_smogn_smoter_comparison.py` computes paired comparison metrics. Git ignores generated result directories.

脚本可能需要以下数据文件：`coal_Q.csv`、`coal_ash.csv`、`coal_V.csv`、`diesel_CN.csv`、`diesel_FREEZE.csv`、`soil_SOC.csv`、`Octane.csv` 和 `Soil.xlsx`。主全光谱实验使用 coal Q、coal ash、diesel CN、diesel FREEZE 和 soil SOC。 / Scripts may require these input files: `coal_Q.csv`, `coal_ash.csv`, `coal_V.csv`, `diesel_CN.csv`, `diesel_FREEZE.csv`, `soil_SOC.csv`, `Octane.csv`, and `Soil.xlsx`. The main full-spectrum protocol uses coal Q, coal ash, diesel CN, diesel FREEZE, and soil SOC.

## 基本调用 / Basic API usage

```python
from respond_spectra import ResponseDrivenAugmenter, load_spectrum_csv, snv

wavelengths, X, y = load_spectrum_csv("data/coal_Q.csv")
X_pre = snv(X)
augmenter = ResponseDrivenAugmenter(n_synthetic=200, random_state=42)
result = augmenter.fit_resample(X_pre, y)

X_aug = result.X
y_aug = result.y
synthetic_mask = result.synthetic_mask
```

`synthetic_mask` 标记生成样本，便于在后续评估中避免其进入验证集或测试集。上例仅演示 API 用法，不代表主实验参数。 / `synthetic_mask` marks generated samples so evaluation can keep them out of validation and test sets. This API example is illustrative and does not specify the main experiment settings.
