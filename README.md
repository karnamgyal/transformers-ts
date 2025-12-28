# Transformer TS Framework (Self-Supervised)

### Description

This project builds a reusable Transformer-based framework for learning representations from long time-series data using self-supervised learning.
The goal is to gain a deeper understanding of Transformers and how self-supervised objectives can be used to learn general representations without labeled data.
The framework is designed to work across domains and can be reused for many downstream machine learning tasks.

### Framework Overview

The framework:
- Takes long time-series signals as input
- Splits signals into fixed-length patches
- Uses a Transformer encoder to model temporal structure
- Trains using masked reconstruction instead of labels
The learned representations are general and reusable.

### Learning Objective

- Understand how Transformers model sequential data
- Apply self-supervised learning in practice
- Learn transferable time-series representations

### Reference Paper

- [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730)

Notes:
- Treats time-series as token sequences
- Uses attention to capture long-range dependencies
- Applies masking for self-supervised training

### Downstream Tasks
The learned embeddings are used for:
- Classification
- Regression
- Forecasting
- Anomaly detection
Embeddings can be frozen or fine-tuned depending on the task.

### Framework Use
This project is structured as a general framework rather than a single experiment.
It is intended for reuse in research, prototyping, and applied ML pipelines.

### Status
Doing literature review on paper and transformer architecture.
Targeted completion: February 2026.
