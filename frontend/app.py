import streamlit as st
import requests


API_URL = "http://localhost:8000/upload-resume/"

st.set_page_config(page_title="Agentic Job Matcher", layout="centered")
st.title("📄 Agentic RAG: Resume Ingestion")
st.write("Upload your resume to generate dense vector embeddings and store them in the local VectorDB.")

uploaded_file = st.file_uploader("Choose a PDF Resume", type="pdf")

if uploaded_file is not None:
    if st.button("Process Resume"):
        with st.spinner("Uploading and processing via LangGraph..."):
            try:
                # Send the file to the FastAPI backend
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Success! {data['status']}")
                else:
                    st.error(f"Failed to process. Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the backend. Is FastAPI running?")

