import streamlit as st
from ngram_model import NGramModel

# Page config
st.set_page_config(page_title="N-Gram NLP App", layout="centered")

# Load data
with open("data.txt") as f:
    sentences = f.readlines()

# Train models
bigram = NGramModel(2)
trigram = NGramModel(3)

bigram.train(sentences)
trigram.train(sentences)

# UI
st.title("Next Word Prediction using N-Grams")
st.write("Bigram & Trigram language model")

user_input = st.text_input("Enter a sentence:")

if user_input:
    st.markdown("---")

    # 🔮 Predictions
    st.subheader(" Next Word Predictions")

    col1, col2 = st.columns(2)

    with col1:
        bigram_pred = bigram.predict_next(user_input)
        st.markdown(
            f"""
            <div style="background-color:#E8F5E9;
                        padding:20px;
                        border-radius:12px;
                        text-align:center;">
                <div style="font-size:18px;color:#2E7D32;">
                    Bigram Prediction
                </div>
                <div style="font-size:42px;font-weight:bold;color:#1B5E20;">
                    {bigram_pred}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        trigram_pred = trigram.predict_next(user_input)
        st.markdown(
            f"""
            <div style="background-color:#E3F2FD;
                        padding:20px;
                        border-radius:12px;
                        text-align:center;">
                <div style="font-size:18px;color:#0D47A1;">
                    Trigram Prediction
                </div>
                <div style="font-size:42px;font-weight:bold;color:#08306B;">
                    {trigram_pred}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # 📊 Probabilities
    st.subheader("Next Word Probabilities")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Bigram Probabilities")
        bigram_probs = bigram.predict_with_probabilities(user_input)

        if bigram_probs:
            for word, prob in bigram_probs.items():
                st.markdown(
                    f"""
                    <div style="background-color:#E8F5E9;
                                padding:10px;
                                margin:6px;
                                border-radius:8px;
                                font-size:18px;">
                        <b>{word}</b> : {prob:.2f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("No bigram probability data")

    with col4:
        st.markdown("### Trigram Probabilities")
        trigram_probs = trigram.predict_with_probabilities(user_input)

        if trigram_probs:
            for word, prob in trigram_probs.items():
                st.markdown(
                    f"""
                    <div style="background-color:#E3F2FD;
                                padding:10px;
                                margin:6px;
                                border-radius:8px;
                                font-size:18px;">
                        <b>{word}</b> : {prob:.2f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("No trigram probability data")
