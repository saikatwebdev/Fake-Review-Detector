# 🕵️‍♂️ Fake Review Detector

This project is an AI-powered web application that detects whether a product review is **authentic** or **suspicious**. Built using `Streamlit`, `scikit-learn`, `NLTK`, and other NLP tools, it analyzes text reviews based on linguistic patterns, sentiment, readability, and other key features.

---

## 🚀 Features

- 📌 **Instant Review Analysis** – Just paste a review, click analyze, and get results instantly.
- 🎯 **Authenticity Detection** – Uses a trained ML model to detect suspicious reviews.
- 📊 **Confidence Scoring** – Outputs a confidence percentage for each prediction.
- 🧠 **Feature Insights** – Breaks down linguistic and structural features like:
  - Sentiment score
  - Readability index
  - Use of capital letters or superlatives
  - Word diversity, grammar patterns, and more
- ⚠️ **Risk Indicators** – Highlights suspicious traits (e.g. extreme sentiment, fake enthusiasm).
- 🔍 **No Data Storage** – Fully privacy-focused; reviews are analyzed in memory.

---

## 🛠️ Tech Stack

- **Frontend & Hosting**: Streamlit
- **ML Model**: Random Forest Classifier (`scikit-learn`)
- **NLP Tools**: NLTK, TextStat
- **Language**: Python 3

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/saikatwebdev/fake-review-detector.git
cd fake-review-detector

```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the app
```bash
streamlit run app.py
```
### 4. Open in browser
Open your browser and go to `http://localhost:8501` to access the app.
### 5. Analyze Reviews
Paste a product review into the input box and click "Analyze". The app will display whether the review is authentic or suspicious, along with detailed feature insights and confidence scores.
---
## 📸 Screenshots

![Screenshot 1](Screenshots\Fake-review-detector.png)
![Screenshot 2](Screenshots\fake-review-detector.png)
