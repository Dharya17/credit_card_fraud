# 💳 Credit Card Fraud Detection

## 🚀 Overview
This project uses **Machine Learning** to detect fraudulent credit card transactions based on key features extracted from transaction data. The model is integrated with a **Streamlit web application** for easy user interaction.

## 📂 Project Structure
```
├── dataset/                   # Contains the dataset
│   ├── creditcard.csv         # Credit card transactions dataset
├── models/                    # Stores trained models
│   ├── fraud_detection_model.pkl  # Trained fraud detection model
│   ├── feature_imputer.pkl    # KNN imputer for missing features
├── app.py                     # Streamlit app for fraud detection
├── train_model.py             # Training script for fraud detection model
├── requirements.txt           # Dependencies for running the project
├── README.md                  # Project documentation (this file)
```

---

## 📊 Dataset Information
- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Fraud Cases**: 492 (Highly imbalanced dataset)
- **Features**:
  - `Amount`: Transaction amount ($)
  - `Time`: Transaction time (seconds since first transaction)
  - `V1 - V28`: PCA-transformed features (confidential banking data)
  - `Class`: (0 = Legitimate, 1 = Fraudulent)

---

## 📌 Installation
1️⃣ **Clone this repository**
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2️⃣ **Create a virtual environment & install dependencies**
```bash
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

3️⃣ **Run the Streamlit app**
```bash
streamlit run app.py
```

---

## 🏗️ Model Training
To train the fraud detection model, run:
```bash
python train_model.py
```
This script:
- Loads and preprocesses the dataset
- Trains a Logistic Regression model
- Saves the model as `fraud_detection_model.pkl`

---

## 🎨 Streamlit Web App
The **Streamlit web application** allows users to **enter transaction details** and check for fraud.

### 🌟 **Features**
✅ User-friendly interface
✅ Accepts key transaction details (amount, risk factors, account trust, etc.)
✅ Uses **KNN Imputer** to fill missing values
✅ Predicts fraud in real-time using trained ML model

### 🖥️ **Run Locally**
```bash
streamlit run app.py
```

### 🌐 **Deploy on Streamlit Cloud**
1️⃣ Push your repository to GitHub
2️⃣ Go to [Streamlit Cloud](https://share.streamlit.io/)
3️⃣ Deploy by connecting your GitHub repo

---

## 📊 Confusion Matrix & Model Performance
The model achieves **high accuracy** while minimizing false negatives (missed fraud cases).

```
            Predicted
            0    |    1
Actual  -----------------
   0    |  TN   |   FP
   1    |  FN   |   TP
```

- **False Negatives (FN)** = 15 (Needs reduction)
- **Strategies to improve**:
  - Use **SMOTE** for class balancing
  - Try **XGBoost** for better fraud detection
  - Tune **threshold values** for fraud classification

---

## 🛠️ Technologies Used
- **Python** (pandas, numpy, scikit-learn, joblib)
- **Machine Learning** (Logistic Regression, KNN Imputer)
- **Streamlit** (For UI & deployment)
- **Jupyter Notebook** (For data exploration)

---

## 🤝 Contributing
1️⃣ Fork the repository 📌
2️⃣ Create a feature branch (`git checkout -b feature-branch`)
3️⃣ Commit changes (`git commit -m 'Added a new feature'`)
4️⃣ Push to GitHub (`git push origin feature-branch`)
5️⃣ Open a Pull Request 🚀

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 📞 Contact
For questions or collaboration, reach out via:
- 📧 Email: your.email@example.com
- 🔗 LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)

🔥 **Happy Coding!** 🚀


