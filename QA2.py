import streamlit as st
import spacy
import spacy_streamlit

# Load spaCy model
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

# Knowledge Base
knowledge_base = """
Python is a programming language.
It was created by Guido van Rossum.
Python was first released in 1991.
Python supports object-oriented programming.
"""

# QA Function
def answer_question(question):
    q = question.lower().strip()

    if "who created python" in q:
        return "Guido van Rossum"

    elif "when was python released" in q:
        return "Python was first released in 1991."

    elif "what is python" in q:
        return "Python is a programming language."

    elif "does python support oop" in q:
        return "Yes, Python supports Object-Oriented Programming."

    else:
        return "Sorry, I don't know the answer."

# Streamlit UI
st.set_page_config(
    page_title="spaCy QA Bot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Question Answering Bot using spaCy")
st.markdown("---")

# User Question
question = st.text_input(
    "Enter your question",
    placeholder="Example: Who created Python?"
)

if st.button("Get Answer"):
    if question:
        answer = answer_question(question)

        st.subheader("Answer")
        st.success(answer)

        st.subheader("spaCy Analysis of Question")
        doc = nlp(question)

        # Display entities, POS tags, dependencies, etc.
        spacy_streamlit.visualize(
            doc,
            labels=nlp.get_pipe("ner").labels
        )

    else:
        st.warning("Please enter a question.")

st.markdown("---")

# Knowledge Base Analysis
st.subheader("Knowledge Base")

if st.checkbox("Show Knowledge Base"):
    st.write(knowledge_base)

    doc_kb = nlp(knowledge_base)

    st.subheader("Named Entity Recognition")
    spacy_streamlit.visualize_ner(
        doc_kb,
        labels=nlp.get_pipe("ner").labels
    )

    st.subheader("Token Analysis")
    token_data = {
        "Token": [token.text for token in doc_kb],
        "POS": [token.pos_ for token in doc_kb],
        "Lemma": [token.lemma_ for token in doc_kb]
    }

    st.dataframe(token_data)

st.markdown("---")
st.caption("Simple Question Answering System using spaCy and spaCy-Streamlit")