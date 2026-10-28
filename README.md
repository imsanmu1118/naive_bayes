# 朴素贝叶斯分类实验

《概率论与数理统计》探究项目：将二值朴素贝叶斯推广到有限多值特征与多类别情形，并应用于 UCI Dry Bean 干豆数据集。

## 目录结构

```
naive_bayes/
├── README.md                     # 本文件
├── bean_raw.csv                  # 原始连续数据（13611 条，含 partition 标记）
├── bean_train.csv                # 离散化训练集 1（9526 条）
├── bean_train_1000.csv           # 离散化训练集 2（1000 条，训练集 1 的子集）
├── bean_test.csv                 # 离散化测试集（4085 条）
├── experiment.py                 # 主实验代码
├── verify_model.py               # 离散朴素贝叶斯数值校验
```

## 环境要求

- Python 3.10 及以上
- NumPy
- pandas
- scikit-learn
- SciPy

本报告使用的版本：Python 3.12.14、NumPy 2.5.2、pandas 3.0.5、scikit-learn 1.9.0、SciPy（随 scikit-learn 安装）。

## 运行方式

在项目目录下执行：

```bash
python experiment.py
```

可选参数：

```bash
python experiment.py --data-dir /path/to/data --output-dir /path/to/results
```

单独校验离散朴素贝叶斯实现：

```bash
python verify_model.py
```

校验内容包括：与 `sklearn.naive_bayes.CategoricalNB` 的预测/概率一致性、拉普拉斯平滑公式、类别先验、未见过的合法取值以及后验概率归一化。

## 实验内容

1. **有限多值朴素贝叶斯**：实现带拉普拉斯平滑的 `DiscreteNaiveBayes`，支持各特征取值个数不同。
2. **两组离散训练集**：分别用 1000 条和 9526 条训练，在 4085 条测试集上评估，并使用配对 McNemar 检验比较差异。
3. **连续特征**：使用 `StandardScaler + GaussianNB`。
4. **相关性与独立性分析**：计算 Pearson 相关系数、类内相关，并按 `|r| ≥ 0.99` 做去冗余对照实验。
5. **改进模型**：`StandardScaler + 多层感知机（MLP）`，网络结构 16–128–64–7，激活函数 `tanh`，Adam 优化器，早停；超参数只在训练集上用三折交叉验证确定。

## 主要结果

| 模型 | 测试准确率 | 宏平均 F1 |
|:---|---:|---:|
| 离散 NB（1000 条训练） | 83.40% | 0.7890 |
| 离散 NB（9526 条训练） | 83.87% | 0.7994 |
| 高斯 NB | 89.84% | 0.9083 |
| 标准化 + MLP | **92.88%** | **0.9392** |

MLP 的三折交叉验证准确率为 93.15%。所有超参数只在训练集上选择，测试集仅用于最终报告。

## 实验报告

`朴素贝叶斯实验报告.docx` 按项目要求组织：

1. 有限多值朴素贝叶斯推导：贝叶斯公式与维数灾难、条件独立假设、参数估计与分类决策、拉普拉斯平滑；
2. 干豆数据分类：分类问题与训练/测试集作用、1000 条和 9526 条训练集的结果、差异分析；
3. 思考问题：连续特征的高斯朴素贝叶斯、相关性与条件独立假设、去冗余实验；
4. 改进方案：标准化 + MLP 的预处理、网络结构、训练参数、结果对比与原因分析。

## 结果文件

运行 `experiment.py` 后，`results/` 目录包含：

- `metrics.csv`：四个模型的总体指标（正确数、准确率、宏/加权 F1、Wilson 95% 置信区间）
- `per_class_metrics.csv`：逐类别 precision / recall / F1 / support
- `summary.json`：指标、McNemar 检验、MLP 参数、相关性/冗余分析和运行环境
- `test_predictions.csv`：各模型在每条测试样本上的预测，便于复核配对检验

## 说明

- 提供的四个 CSV 文件不应修改；`experiment.py` 会检查行数、缺失值、离散取值、训练/测试划分和 1000 条子集的一致性。
- `bean_raw.csv` 的 TEST 行顺序与 `bean_test.csv` 不同，代码按 `id` 对齐后再评估。
- 随机种子统一为 42，结果可复现。
