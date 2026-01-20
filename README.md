
# 2025-2026学年东北大学机器学习大作业
# Pre-training Time Series Models with Stock Data

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research-green)](https://github.com/)

> **核心理念**：与其让模型直接在充满噪声的股价数据上“猜涨跌”，不如先教它识别“这是哪只股票”。通过“身份识别”预训练，模型能学会捕捉不同股票独特的波动纹理和市场微观结构，从而显著提升下游趋势预测的泛化能力。

## 📖 项目简介 (Introduction)

本项目提出了一种**“先博通，后专精”**的金融时间序列深度学习框架。针对个股历史数据有限、信噪比低导致过拟合的问题，我们设计了一个基于 **Transformer Encoder** 的两阶段迁移学习方案：

1.  **阶段一：自监督身份感知预训练 (Pre-training)**
    *   任务：输入 30 天的 K 线行情，分类其属于哪只股票（N 分类任务）。
    *   目的：强迫模型学习成交量异动、波动率聚类等深层市场特征。
2.  **阶段二：趋势预测微调 (Fine-tuning)**
    *   任务：输入同样的序列，预测次日收盘价的涨跌（二分类任务）。
    *   策略：冻结底层特征提取器，仅微调高层预测头。

实验证明，该方法相比从头训练（Training from Scratch）的基线模型，在 **MCC（马修斯相关系数）** 和 **AUC** 指标上实现了质的突破。

## 🚀 核心特性 (Key Features)

*   **Transformer 架构**：采用 Multi-Head Self-Attention 处理长距离时间序列依赖，替代传统的 LSTM/GRU。
*   **严谨的数据防泄露机制**：
    *   基于时间的训练/测试集切分（TimeBasedSplit）。
    *   动态标准化（Dynamic Normalization）：`scaler` 仅在训练集拟合，应用于测试集，严禁未来信息泄露。
*   **多因子输入**：融合价量信息，特征包括 `Open_pct`, `High_pct`, `Low_pct`, `Close_pct`, `Volume_log`。
*   **Last-Token Pooling**：针对金融数据的时效性，采用末端标记池化策略，最大化保留最新市场状态。
*   **全方位评估指标**：集成 Accuracy, F1-Score, MCC (Matthews Correlation Coefficient), AUC (Area Under ROC Curve)。

## 🛠️ 环境依赖 (Requirements)

本项目基于 Python 3 开发，核心依赖库如下：

```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm akshare
```

*   **PyTorch**: 深度学习框架。
*   **AkShare**: 开源金融数据接口（用于获取美股/A股历史行情）。
*   **Scikit-learn**: 数据预处理与评估指标计算。
*   **Matplotlib**: 训练曲线可视化。

## 🏃‍♂️ 快速开始 (Quick Start)

1.  **克隆仓库**：
    ```bash
    git clone https://github.com/9Bingo/Stock-TS-Pretrain.git
    ```

2.  **运行主程序**：
    该脚本会自动下载数据、处理数据、执行预训练、微调、基线对比并绘图。
    ```bash
    python main.py
    ```
    *(注：首次运行需要下载数据，请确保网络连接正常，且能访问 AkShare 数据源)*

## 📊 实验结果 (Results)

基于美股科技、金融等板块 60 支龙头股（2016-2024）的实测数据：

| 模型架构 | Accuracy | F1 Score | **MCC (核心指标)** | AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (无预训练)** | 50.94% | 0.6499 | -0.0105 (失效) | 0.4927 |
| **Ours (预训练微调)** | **51.24%** | 0.5551 | **0.0199 (有效)** | **0.5141** |

*   **MCC 反转**：从负值（差于随机）变为正值（具备预测力），证明预训练学到了鲁棒特征。
*   **AUC 提升**：突破 0.5 阈值，表明模型输出的概率具有排序交易价值。

## 📂 代码结构 (Code Structure)

*   **`Config` 类**:
    *   统一管理超参数（时间窗口 `T=30`，隐层维度 `d_model=64`，学习率等）。
*   **`StockDataset` 类**:
    *   实现滑动窗口切片，将连续的时间序列转化为 `(Sample, Time, Feature)` 张量。
*   **`StockTransformer` 类**:
    *   核心模型。支持通过 `cls_type='id'` 或 `cls_type='price'` 切换任务头。
    *   包含 `PositionalEncoding` 和 `TransformerEncoderLayer`。
*   **`train_phase1_identity`**:
    *   执行股票身份识别预训练任务。
*   **`train_phase2_finetune`**:
    *   加载预训练权重，冻结部分层，执行价格预测微调。
*   **`analyze_results`**:
    *   绘制 Loss, Acc, F1, MCC, AUC 五维对比图表，并保存为 `comprehensive_analysis.png`。
