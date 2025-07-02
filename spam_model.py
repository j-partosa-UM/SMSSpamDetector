import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


class SpamClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self.trained = False

    def load_and_prepare_data(self, path: str):
        df = pd.read_csv(path, encoding='latin-1')[['v1', 'v2']]
        df.columns = ['label', 'text']
        df['text'] = df['text'].str.lower()
        return df

    def train(self, df: pd.DataFrame, test_size: float = 0.3):
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['label'], test_size=test_size, random_state=42
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.model.fit(X_train_vec, y_train)
        self.trained = True

        y_pred = self.model.predict(X_test_vec)

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred)
        }

    def predict(self, text: str) -> str:
        if not self.trained:
            raise Exception("Model not trained yet.")
        vec = self.vectorizer.transform([text.lower()])
        return self.model.predict(vec)[0]