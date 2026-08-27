

# Emotion Classification with RNNs, LSTMs & GRUs

A deep learning-based **Natural Language Processing (NLP)** project that classifies text into six different emotions using recurrent neural network architectures.

The project compares foundational sequence models such as **Simple RNN, LSTM, and GRU**, before progressing toward an advanced **Bidirectional GRU** architecture.

## 🎯 Project Objective

The goal of this project is to build a text classification system capable of understanding the emotional context of a sentence and predicting its corresponding emotion.

The model classifies text into:

| Label | Emotion  |
| ----: | -------- |
|     0 | Sadness  |
|     1 | Joy      |
|     2 | Love     |
|     3 | Anger    |
|     4 | Fear     |
|     5 | Surprise |

The dataset contains **16,000 training samples** and **2,000 test samples**.  

## 🧠 Models Implemented

The project explores multiple recurrent architectures:

* Simple RNN
* LSTM
* GRU
* Bidirectional GRU

The purpose of comparing these architectures is to understand how different recurrent networks handle sequential text information.

## 📊 Dataset

This project uses the **Emotion Dataset** from Hugging Face:

`dair-ai/emotion`

The dataset contains six emotion categories:

* 😢 Sadness
* 😄 Joy
* ❤️ Love
* 😡 Anger
* 😨 Fear
* 😲 Surprise

The training data is somewhat imbalanced. For example, the training set contains 5,362 `joy` samples, while `surprise` contains 572 samples. 

To address this imbalance, **balanced class weights** are calculated during training. 

## 🔄 NLP Preprocessing Pipeline

The text goes through the following preprocessing pipeline:

```text
Raw Text
   ↓
Tokenization
   ↓
Word → Integer IDs
   ↓
Sequence Padding
   ↓
Embedding Layer
   ↓
RNN / LSTM / GRU
   ↓
Dense Output Layer
   ↓
Emotion Prediction
```

The project uses the Keras `Tokenizer` with:

* Vocabulary limit: `10,000`
* OOV token: `<unk>`
* Maximum sequence length: `50`



## 🧩 Embedding Layer

The tokenized words are passed into an **Embedding Layer**.

Instead of representing every word as a large sparse one-hot vector, the embedding layer represents each token as a dense numerical vector.

Conceptually:

```text
"happy"
   ↓
Token ID
   ↓
Embedding Layer
   ↓
[0.21, -0.14, 0.72, ...]
```

These vectors allow the neural network to learn relationships between words based on their usage in the training data.

## ⚖️ Handling Class Imbalance

Since the dataset contains significantly different numbers of examples for each emotion, balanced class weights are calculated using:

```python
class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_label),
    y=train_label
)
```

This helps prevent the model from becoming overly biased toward frequently occurring classes. 

## 🛑 Early Stopping

The project uses **EarlyStopping** to reduce overfitting.

The training process monitors:

```text
validation loss
```

and stops training when the validation performance stops improving.

The configuration uses:

```python
EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
```



## 🏗️ Project Architecture

Recommended production structure:

```text
emotion-classification/
│
├── app/
│   ├── main.py
│   ├── schemas.py
│   │
│   └── model/
│       ├── emotion_model.keras
│       └── tokenizer.pkl
│
├── notebooks/
│   └── Predictiong_emotions_of_text.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 💾 Model Deployment

For deployment, the trained model and tokenizer should be saved separately.

### Save the model

```python
model.save("emotion_model.keras")
```

### Save the tokenizer

```python
import pickle

with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)
```

The tokenizer is essential because the same word-to-index mapping used during training must be used during inference.

## 🚀 FastAPI Inference

The trained model can be exposed through a REST API using **FastAPI**.

Example request:

```json
{
    "text": "I am extremely happy today!"
}
```

Example response:

```json
{
    "emotion": "joy",
    "confidence": 0.94
}
```

The inference pipeline is:

```text
Client
  ↓
FastAPI
  ↓
Receive Text
  ↓
Tokenizer
  ↓
Sequence
  ↓
Padding
  ↓
Trained Neural Network
  ↓
Softmax Probabilities
  ↓
Predicted Emotion
  ↓
Confidence Score
```

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras
* Hugging Face Datasets
* FastAPI
* Pickle
* Jupyter / Google Colab

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yoloaryan/Predicting_emotions_of_text.git
cd Predicting_emotions_of_text
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Notebook

Open the notebook:

```text
notebooks/Predictiong_emotions_of_text.ipynb
```

The notebook covers:

1. Library installation/import
2. Dataset loading
3. Exploratory Data Analysis
4. NLP preprocessing
5. Tokenization
6. Sequence padding
7. Class-weight calculation
8. RNN training
9. LSTM training
10. GRU training
11. Bidirectional GRU
12. Model evaluation

## 🌐 Running FastAPI

After saving the trained model and tokenizer:

```bash
uvicorn app.main:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

You can then test the `/predict` endpoint directly through Swagger UI.

## 📈 Future Improvements

Potential improvements for the project include:

* Transformer-based emotion classification
* BERT / DistilBERT implementation
* Hyperparameter tuning
* Better handling of class imbalance
* Model explainability
* Confidence visualization
* Docker containerization
* Cloud deployment
* Production monitoring
* Frontend interface for real-time predictions

## 👨‍💻 Author

**Aryan**

GitHub: `yoloaryan`

---

### ⭐ Project Highlights

> **NLP + Deep Learning + RNN/LSTM/GRU + FastAPI + Deployment**

This makes the project more than a basic notebook: it demonstrates the complete workflow from **dataset → preprocessing → model training → evaluation → API inference → deployment**.

