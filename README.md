# SMSSpamDetector
# 📧 Spam Message Classifier (Streamlit App)

A simple web app built with **Python**, **scikit-learn**, and **Streamlit** that classifies SMS messages as **Spam** or **Ham** using a Naive Bayes model trained on the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

---

## 🖼️ Screenshot

![App Screenshot](app-screenshot(1).png)
![App Screenshot](app-screenshot(2).png)

> Replace `screenshot.png` with your actual screenshot file in the project directory.

---

## ⚙️ Features

- Load and clean the spam dataset
- Train a `MultinomialNB` classifier using `CountVectorizer`
- Show accuracy and classification report
- Classify custom user messages via web UI

---

## 🚀 How to Run

### 1. Clone the repository or download the code

```bash
git clone https://github.com/j-partosa-UM/SMSSpamDetector.git
cd spam-classifier-app

