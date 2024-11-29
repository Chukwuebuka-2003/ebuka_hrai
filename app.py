import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Google Generative AI Setup
api_key = "AIzaSyCmt0_tTBH65Zq7LpNFk-TMeSXR_UTh_zI"  # Replace with your API key
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)

# Define Prompt Template
prompt_template = PromptTemplate(
    input_variables=["resume_text"],
    template=(
        "You are a professional resume reviewer. Provide detailed feedback on the following resume:\n\n"
        "{resume_text}\n\n"
        "Your feedback should include suggestions on structure, language, skills, and overall presentation."
    ),
)

# Create LangChain LLM Chain
resume_review_chain = LLMChain(llm=model, prompt=prompt_template)

# Streamlit UI
st.title("AI Resume Reviewer")
st.write("Upload your resume, and the AI will provide detailed feedback.")

# File uploader
uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file:
    # Extract text from uploaded file
    file_type = uploaded_file.type
    resume_text = ""
    
    if file_type == "application/pdf":
        try:
            reader = PdfReader(uploaded_file)
            resume_text = "\n".join(page.extract_text() for page in reader.pages)
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
    elif file_type == "text/plain":
        resume_text = uploaded_file.read().decode("utf-8")
    
    if resume_text.strip():
        # Display uploaded resume content
        st.subheader("Uploaded Resume Text")
        st.text_area("Resume Content", value=resume_text, height=300)
        
        # AI Feedback
        st.subheader("AI Feedback")
        with st.spinner("Analyzing your resume..."):
            try:
                feedback = resume_review_chain.run({"resume_text": resume_text})
                st.write(feedback)
            except Exception as e:
                st.error(f"Error generating feedback: {e}")
    else:
        st.error("Could not extract text from the uploaded file. Please try again.")
