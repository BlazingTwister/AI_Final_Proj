import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load model and tokenizer from the saved directory
model = GPT2LMHeadModel.from_pretrained('saved_model')
tokenizer = GPT2Tokenizer.from_pretrained('saved_model')

# Define available question types
question_types = [
    "General Knowledge for Kids",
    "GK Questions for Class 1",
    "GK Questions for Class 2",
    "GK Questions for Class 3",
    "GK Questions for Class 4",
    "GK Questions for Class 5",
    "GK Questions for Class 6",
    "GK Questions for Class 7"
]

# Streamlit app title
st.title("Question Generator App")

# Dropdown menu for question type
selected_question_type = st.selectbox("Select Question Type:", question_types)

# Function to generate question
def generate_question(question_type):
    input_ids = tokenizer.encode(question_type, return_tensors='pt', padding=True, truncation=True)
    output_ids = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2)
    generated_question = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_question

# Generate and display the question on button click
if st.button("Generate Question"):
    generated_question = generate_question(selected_question_type)
    st.success(f"Generated Question: {generated_question}")
