# 💳 Fraud Detection App

A machine learning-powered web application that predicts whether a financial transaction is **fraudulent or legitimate** — built with Python, scikit-learn, and Streamlit.

---

## 📌 Overview

This project walks through the complete data science pipeline — from exploratory data analysis (EDA) to model training and deployment — using a real-world financial transactions dataset. The final product is an interactive web app where users can input transaction details and get an instant fraud prediction.

---

## 📁 Project Structure

```
fraud-detection/
│
├── analysis_model.ipynb          # EDA + model training notebook
├── fraud_detection_pipeline.pkl  # Saved trained model pipeline
├── fraud_detection.py            # Streamlit web application
├── AIML Dataset.csv              # Dataset (download from Kaggle)
└── README.md
```

---

## 📊 Dataset

**Source:** [Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download)

> 📥 Download the dataset from the link above and place `AIML Dataset.csv` in the project root before running the notebook.

The dataset contains financial transaction records with the following key features:

| Feature | Description |
|---|---|
| `type` | Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEPOSIT) |
| `amount` | Transaction amount |
| `oldbalanceOrg` | Sender's balance before the transaction |
| `newbalanceOrig` | Sender's balance after the transaction |
| `oldbalanceDest` | Receiver's balance before the transaction |
| `newbalanceDest` | Receiver's balance after the transaction |
| `isFraud` | Target label — 1 = Fraud, 0 = Legitimate |

---

## 🧠 Model & Approach

### Exploratory Data Analysis (EDA)
- Distribution of transaction types
- Fraud rate by transaction type
- Log-scaled amount distribution
- Correlation heatmap
- Fraud patterns over time
- Balance difference analysis (zero-out-after-transfer pattern)

### Preprocessing Pipeline
- **Numerical features:** StandardScaler
- **Categorical features:** OneHotEncoder (drop first)
- Built using `sklearn.pipeline.Pipeline` + `ColumnTransformer`

### Model
- **Algorithm:** Logistic Regression (`class_weight="balanced"`, `max_iter=1000`)
- **Train/Test Split:** 70/30 with stratification
- **Evaluation:** Classification report + Confusion matrix
- Model saved using `joblib`

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
```

### 2. Install dependencies
```bash
pip install streamlit pandas scikit-learn joblib
```

### 3. Download the dataset
Download `AIML Dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download) and place it in the project root.

### 4. Train the model (optional — `.pkl` already included)
Open and run `analysis_model.ipynb` in Jupyter Notebook to regenerate the model pipeline.

### 5. Run the Streamlit app
```bash
streamlit run fraud_detection.py
```

---

## 📦 Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- joblib

---

## 📈 Results

The Logistic Regression model is trained on a heavily imbalanced dataset and handles class imbalance via `class_weight="balanced"`. Evaluation metrics include precision, recall, F1-score, and a confusion matrix.

---

## 🙏 Acknowledgements

- Dataset by [Aman Ali Siddiqui on Kaggle](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
