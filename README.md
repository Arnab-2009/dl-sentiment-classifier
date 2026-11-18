# dl-sentiment-classifier
# 🧠 Deep Learning Customer Review Sentiment Classifier (PyTorch)

An end-to-end **Natural Language Processing (NLP)** and **Deep Learning** pipeline built in **PyTorch** to classify customer product reviews into Positive or Negative sentiment in real time.

---

### 🎯 Use Case
E-commerce stores and SaaS teams receive thousands of customer reviews and support tickets daily. Automated sentiment classification flags negative feedback immediately so support teams can prioritize urgent issues and track product satisfaction trends.

---

### 🏗️ Neural Network Architecture
1. **Custom Tokenizer & Vocabulary Builder:** Cleans raw text via Regex, maps tokens to integer IDs with `<PAD>` and `<UNK>` handling, and pads sequences to fixed length.
2. **Embedding Layer (`nn.Embedding`):** Learns dense vector representations for each word token.
3. **Global Average Pooling:** Aggregates token embeddings across the sequence length into a fixed-size semantic representation.
4. **Dense Classification Head:** `Linear -> ReLU -> Dropout(0.3) -> Linear` trained with **Adam** optimizer and **Binary Cross-Entropy with Logits Loss (`BCEWithLogitsLoss`)**.

---

### 📈 Training Dynamics & Evaluation
Running the script automatically generates training/validation loss and accuracy curves alongside a test-set confusion matrix:

| Training & Validation Curves | Confusion Matrix |
| :---: | :---: |
| ![Training Curves](outputs/training_curves.png) | ![Confusion Matrix](outputs/confusion_matrix.png) |

---

### 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Arnab-2009/dl-sentiment-classifier.git](https://github.com/Arnab-2009/dl-sentiment-classifier.git)
   cd dl-sentiment-classifier
