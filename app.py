import streamlit as st
from spam_model import SpamClassifier

st.set_page_config(page_title="Spam Classifier", layout="centered")

st.title("📧 Spam Message Classifier")

# initialize classifier and train

def load_model():
    clf = SpamClassifier()
    df = clf.load_and_prepare_data("spam.csv")
    metrics = clf.train(df)
    return clf, metrics

model, metrics = load_model()

# show model performance
st.subheader("Model Performance")
st.write(f"**Accuracy:** {metrics['accuracy']:.4f}")
with st.expander("See classification report"):
    st.code(metrics["report"], language="text")

# user input
st.subheader("Try Your Own Message")
user_input = st.text_area("Enter a message to classify:", height=100)

if st.button("Classify"):
    if user_input.strip():
        result = model.predict(user_input)
        if result == "spam":
            st.error("This message is classified as **SPAM**")
        else:
            st.success("This message is classified as **HAM** (not spam).")
    else:
        st.warning("Please enter a message.")