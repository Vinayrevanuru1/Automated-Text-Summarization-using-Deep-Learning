import streamlit as st
import pandas as pd
import pickle
from simplet5 import SimpleT5

from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# Replace 'your_local_directory' with the path to your saved model
# model_name = "facebook/bart-large-cnn"  # Replace with the actual model name on Hugging Face
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# summarizer = pipeline('summarization', model=model, tokenizer=tokenizer)

model_T5 = SimpleT5()
model_T5.load_model("t5","model14epoch", use_gpu=True)

def summarize_text_T5(text):
    return model_T5.predict(text)

def summarize_text_bart(text):
    return model_T5.predict(text)

summarized_text = []

st.header('Text Summarizer')
model_selct = st.sidebar.selectbox('Select Model', ["T5", "Bart-CNN"])
text = st.text_area('Enter Article')
st.write(f"Text Length - {len(text.split(' '))}")
button_pressed = st.button('Summarize')
if len(text) != 0 and button_pressed:
    if model_selct == "T5":
        summarized_text = summarize_text_T5(text) 
    # elif model_selct == "Bart-CNN":
    #     summarized_text = summarizer(text, max_length=100)
    else:
        summarized_text = summarize_text_bart(text) 
        
elif len(text) == 0 and button_pressed:
    summarized_text = "Enter Text"
if len(summarized_text) != 0:
    st.write(f"Text Length - {len(summarized_text[0].split(' '))}")
    st.write(summarized_text[0])