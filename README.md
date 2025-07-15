
# Accelerating DNA Sequence Alignment Using Optimized AI Techniques

## Overview

This repository presents a machine learning-based framework to accelerate DNA sequence alignment by replacing computationally expensive traditional algorithms with optimized artificial intelligence techniques. The project implements and compares the performance of Convolutional Neural Networks (CNN), Bidirectional Long Short-Term Memory networks (Bi-LSTM), and Random Forest (RF) classifiers for high-accuracy and low-latency sequence alignment.

---

## Key Features

- Implements CNN, Bi-LSTM, and Random Forest models for classification of DNA sequence alignments
- Achieves up to **99.97% accuracy** using CNN with reduced training time
- Provides a balanced trade-off between accuracy, speed, and hardware requirements
- Suitable for large-scale genomic applications like disease diagnosis, drug discovery, and evolutionary analysis

---

## Problem Statement

Traditional sequence alignment algorithms such as Needleman-Wunsch (NW) are mathematically optimal but computationally expensive, especially for long sequences or large datasets. They exhibit a time complexity of \(O(MN)\), where \(M\) and \(N\) are the lengths of the sequences to be aligned. This project explores how machine learning models can overcome these limitations by learning alignment patterns in advance, thereby reducing computational time and making the process scalable for real-time applications.

---

## Models Implemented

### 🔹 Convolutional Neural Network (CNN)
- Accuracy: **99.97%**
- Optimized with ADAM optimizer and dropout regularization
- Efficient feature extraction on encoded DNA inputs

### 🔹 Bidirectional LSTM (Bi-LSTM)
- Accuracy: **99.96%**
- Captures long-range dependencies in DNA sequences
- Requires higher training time compared to CNN

### 🔹 Random Forest (RF)
- Accuracy: **99.80%**
- Minimal training time
- Best suited for environments with low computational resources

---

## Dataset

- Source: [IEEE DataPort – DNA Sequence Alignment Dataset](https://ieee-dataport.org/documents/dna-sequence-alignment-datasets-based-nw-algorithm)
- Files used:
  - `RefSeq.csv` – Used for training
  - `CompSeq.csv` – Used for testing
- Each sample consists of:
  - 16 input features (fr1 to fr16), representing encoded nucleotide pairs
  - 1 output class indicating the alignment result
- Total classes: 254 (unique global alignments)

---

## Project Structure

```
📁 root/
│
├── CNN.ipynb                  # CNN-based sequence alignment model
├── Bi-LSTM.ipynb              # Bi-LSTM-based sequence alignment model
├── Random-Forest.ipynb        # Random Forest classifier
├── Random-Forest_single-file.ipynb
├── MLP.ipynb                  # Baseline MLP model
├── Results.ipynb              # Performance comparison and evaluation
├── Dataset/                   # RefSeq.csv and CompSeq.csv
└── README.md                  # This file
```

---

## Evaluation Metrics

| Model         | Accuracy | Precision | Recall | F1 Score | Training Time | Testing Time |
|---------------|----------|-----------|--------|----------|----------------|---------------|
| CNN           | 99.97%   | 99.96%    | 99.97% | 99.97%   | 1604s          | 11.26s        |
| Bi-LSTM       | 99.96%   | 99.96%    | 99.76% | 99.83%   | 4808s          | 12.34s        |
| Random Forest | 99.80%   | 99.81%    | 99.80% | 99.80%   | 237.6s         | 12.28s        |
| MLP (Baseline)| 99.70%   | 99.70%    | 99.70% | 99.70%   | 4093s          | 6.04s         |

---

## Installation

Ensure you have Python ≥ 3.8 installed. Then install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras
```

Place the datasets (`RefSeq.csv` and `CompSeq.csv`) inside a folder named `Dataset` in the root directory.

---

## Usage

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/dna-sequence-alignment-ai.git
cd dna-sequence-alignment-ai
```

2. **Run the notebooks**

Open any of the following notebooks using Jupyter or your preferred environment:

- `CNN.ipynb` – Runs the CNN-based DNA alignment model
- `Bi-LSTM.ipynb` – Runs the Bi-LSTM model
- `Random-Forest.ipynb` – Runs the Random Forest classifier
- `MLP.ipynb` – Baseline MLP model
- `Results.ipynb` – Evaluation and comparison of all models

Make sure the dataset is loaded correctly and paths are adjusted if needed.

---

## Future Work

- Build a web-based tool for real-time DNA sequence alignment
- Extend the approach to support RNA and protein sequence data
- Integrate optimization techniques like Genetic Algorithms or Particle Swarm Optimization
- Accelerate models using Numba, PyPy, or convert to tensor ops with Hummingbird-ML
- Deploy model on cloud-based or FPGA/edge computing platforms for scalable analysis

---

## Authors

- **K Supriya**
- **R Kavin Kumar**

Department of Electronics and Communication Engineering  
Amrita Vishwa Vidyapeetham, Bengaluru Campus

---

## License

This project is intended for academic and research purposes only.  
For reuse or publication, please contact the authors directly.
