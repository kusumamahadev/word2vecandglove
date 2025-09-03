# 📊 AI-Driven Sentiment Classification for E-Commerce  

## 🌟 Overview  
This project builds a predictive sentiment classification model for an e-commerce platform specializing in **electronic gadgets**.  

With a **200% increase in customer base** and a **25% spike in feedback volume**, manual review is no longer feasible.  
The solution leverages **NLP + Machine Learning** to classify customer reviews as **Positive, Negative, or Neutral**.  

---

## 📂 Dataset  

| Column Name      | Type       | Description |
|------------------|-----------|-------------|
| **Product ID**   | String/Int | Unique identifier for each product |
| **Product Review** | Text     | Customer feedback and opinions |
| **Sentiment**    | Categorical | Positive / Negative / Neutral |

---

## 🛠 Tech Stack  

- **Python 3.9+**  
- **Libraries**:  
  - Data Processing → `pandas`, `numpy`  
  - NLP → `nltk`, `spacy`, `scikit-learn`  
  - Modeling → `Logistic Regression`, `Random Forest`, `Naive Bayes`, `XGBoost`  
  - Visualization → `matplotlib`, `seaborn`  
- **Optional**: `TensorFlow`, `PyTorch`  

---

## 🔎 Approach  

1. **Preprocessing** → Clean text, tokenize, lemmatize, vectorize (TF-IDF / Word2Vec)  
2. **EDA** → Sentiment distribution, frequent words  
3. **Modeling** → Train & evaluate ML models  
4. **Evaluation** → Accuracy, Precision, Recall, F1-score  
5. **Future Scope** → Deploy API (Flask/FastAPI), dashboards  

---

## 📈 Evaluation Metrics  

- **Accuracy** → Overall performance  
- **Precision** → % predicted positives that are correct  
- **Recall** → % of actual positives correctly identified  
- **F1-Score** → Balance of Precision & Recall  
- **Confusion Matrix** → Visual analysis  

---

## 🚀 Getting Started  

 Clone Repo  
```bash
git clone https://github.com/your-username/word2vecandglove.git
cd word2vecandglove

 Install Dependencies
pip install -r requirements.txt
 Run Notebook / Script
jupyter notebook Sentiment_Analysis.ipynb
# OR
python train_model.py
