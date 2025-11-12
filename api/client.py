import requests
import streamlit as st

def get_response(input_text):
    response = requests.post('http://localhost:8000/concept-explanation/invoke',
     json={
        'input':{
            'text': input_text
        }})
    data= response.json()
    return data['output']['content']
def get_summary(input_text2):
    response = requests.post('http://localhost:8000/text-summarization/invoke',
     json={
        'input':{
            'text': input_text2
        }})
    return response.json()['output']

st.title("Concept Explainer")
input_text = st.text_input("Enter a concept to explain:")
input_text2= st.text_input("Enter text to summarize:")

if st.button("Explain Concept"):
    output = get_response(input_text)
    st.write("Explanation:", output)
if st.button("Summarize Text"):
    output2 = get_summary(input_text2)
    st.write("Summary:", output2)