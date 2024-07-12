import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load the saved model and tokenizer
model_path = 'txt-autocomplete-model.h5'
tokenizer_path = 'tokenizer.pickle'

model = load_model(model_path)

with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to preprocess input and predict
def predict_top_five_words(model, tokenizer, seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=453-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    top_five_indexes = np.argsort(predicted[0])[::-1][:5]
    top_five_words = []
    for index in top_five_indexes:
        for word, idx in tokenizer.word_index.items():
            if idx == index:
                top_five_words.append(word)
                break
    return top_five_words

# Example sentences for demonstration
example_sentences = [
    "it is a great",
    "there was a hidden",
    "Those kids want to",
    "we love our"
]

# Streamlit App
st.title("Text Auto-complete")
st.write("Enter a sequence of text and click the Predict button to see the next word prediction.")


# Prediction functionality
input_text = st.text_area("Enter your text here:", "")

if st.button("Predict"):
    if input_text:
        try:
            next_words = predict_top_five_words(model, tokenizer, input_text)
            if next_words:
                st.success(f"Predicted next words: {', '.join(next_words)}")
            else:
                st.error("Could not predict the next words. Please try again.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please enter some text to predict.")


# Display examples
st.markdown("### Example Sentences and Predictions:")
for sentence in example_sentences:
    try:
        prediction = predict_top_five_words(model, tokenizer, sentence)
        st.write(f"**Input:** '{sentence}'")
        st.write(f"**Predictions:** {', '.join(prediction)}")
        st.write("---")
    except Exception as e:
        st.error(f"Error predicting for '{sentence}': {e}")


# Adding some style
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f0f5;
        padding: 20px;
        border-radius: 10px;
    }
    .stTextArea textarea {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
