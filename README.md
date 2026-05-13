# Student Name:- Amey Banarase
# Student Id:- bitsom_ba_2511968

# Part 3: NLP and Sequence Modeling Mini Project

## Project Overview
This project builds an NLP pipeline to classify customer support messages into 3 sentiment categories: **positive**, **neutral**, and **negative**. It compares a traditional machine learning baseline (Logistic Regression + TF-IDF) with a deep learning sequence model (LSTM).

## Dataset Source
**Dataset:** Customer Support Text Classification Dataset  
**Source:** BITSoM Module 5 - Part 3 Shared Google Drive Folder  
https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing

## Dataset Description
| Property | Details |
|---|---|
| Total Records | 1500 |
| Input Feature | customer_message (text) |
| Target Variable | sentiment_label |
| Classes | positive, neutral, negative |
| Class Balance | ~500 per class (balanced) |

## Repository Structure
```
part-3-nlp-sequence-modeling/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── class_distribution.png
    ├── baseline_confusion_matrix.png
    ├── lstm_confusion_matrix.png
    ├── model_evaluation.png
    ├── model_evaluation.csv
    └── sample_predictions.txt
```

## Tasks Completed

### Task 1: Dataset Understanding
- 1500 customer support messages across 3 sentiment classes
- Balanced dataset — approximately 500 samples per class
- Average message length analyzed and visualized
- Class distribution plotted

### Task 2: Text Preprocessing
- **Lowercasing:** All text converted to lowercase
- **Special character removal:** Removed punctuation, numbers, symbols
- **Tokenization:** Split text into individual words
- **Stopword removal:** Removed common words (the, is, at, etc.) using NLTK

### Task 3: Text Vectorization
**Method Used: TF-IDF (Term Frequency - Inverse Document Frequency)**
- Vocabulary size: 5000 most important words
- n-gram range: (1, 2) — captures single words and word pairs
- TF-IDF gives higher weight to words that are important in a document but rare across all documents

**Why text must be converted to vectors:**  
Machine learning models only understand numbers. Text must be converted to numerical vectors so mathematical operations (dot products, gradients) can be applied. Without vectorization, a model cannot process raw text.

### Task 4: Baseline Model
**Model: Logistic Regression with TF-IDF**
- Simple, fast, and interpretable
- Good baseline for text classification

### Task 5: Sequence Model - LSTM
**Architecture:**
```
Input: padded sequence (MAX_LEN=50)
Embedding Layer (5000 vocab, 64 dims)
LSTM Layer (64 units)
Dropout (0.3)
Dense (32, ReLU)
Output Dense (3, Softmax)
```
- **Loss:** Categorical Crossentropy
- **Optimizer:** Adam
- **Epochs:** 10

### Task 6: Attention and Transformer Reflection

**Why RNNs Struggle with Long-Term Dependencies:**  
RNNs pass information through a hidden state step by step. For long sentences, early information gets diluted — this is the vanishing gradient problem.

**How LSTMs Help with Memory:**  
LSTMs use three gates (forget, input, output) to selectively remember and forget information, allowing them to capture long-range dependencies.

**What Attention Solves:**  
Attention allows the model to directly focus on relevant parts of the input at each step, avoiding the bottleneck of compressing everything into one vector.

**Why Transformers are Important:**  
Transformers process all words in parallel using self-attention, are highly scalable, and form the foundation of modern NLP models like BERT, GPT, and Claude.

## Model Comparison
| Model | Vectorization | Test Accuracy |
|---|---|---|
| Logistic Regression | TF-IDF | ~85% |
| LSTM | Embedding + Sequences | ~87% |

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Technologies Used
- Python 3.x
- TensorFlow / Keras (LSTM)
- Scikit-learn (Logistic Regression, TF-IDF)
- NLTK (text preprocessing)
- Pandas, NumPy
- Matplotlib, Seaborn
