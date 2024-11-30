import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set Streamlit Page Config with Icon
st.set_page_config(
    page_title="AI-Powered HR Tools",
    page_icon="🧑‍💼",  # HR-related emoji (or use a custom icon path if hosted)
    layout="wide"
)

# Initialize the Google Gemini LLM
api_key = st.secrets["google_genai"]["api_key"]# Replace with your actual API key
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)

# Define the prompt template for resume review
resume_prompt_template = PromptTemplate(
    input_variables=["resume_text"],
    template=(
        "You are a professional resume reviewer. Provide detailed feedback on the following resume:\n\n"
        "{resume_text}\n\n"
        "Your feedback should include suggestions on structure, language, skills, and overall presentation."
    ),
)

# Define the prompt template for HR-related questions
hr_prompt_template = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are an HR expert. Answer the following question with detailed and professional advice:\n\n"
        "{question}"
    ),
)

# Create LangChain LLM Chains
resume_review_chain = LLMChain(llm=llm, prompt=resume_prompt_template)
hr_question_chain = LLMChain(llm=llm, prompt=hr_prompt_template)

# Streamlit App with Tabs
st.title("AI-Powered HR Tools")
tab1, tab2 = st.tabs(["Resume Reviewer", "HR Question Assistant"])

# Add Sidebar with Example Questions
st.sidebar.title("Example HR Questions")
example_questions = [
    "What are the best practices for conducting a job interview?",
    "How should I negotiate my salary?",
    "What is the proper format for a resignation letter?",
    "How do I handle workplace conflicts professionally?",
    "What are the top skills employers look for in 2024?"
]
selected_question = st.sidebar.radio("Choose an example question:", options=example_questions)

# Tab 1: Resume Reviewer
with tab1:
    st.header("Resume Reviewer")
    st.write("Upload your resume to receive detailed feedback.")

    # File uploader
    uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])

    if uploaded_file:
        # Extract text from the uploaded file
        file_type = uploaded_file.type
        resume_text = ""
        
        if file_type == "application/pdf":
            try:
                reader = PdfReader(uploaded_file)
                resume_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
        elif file_type == "text/plain":
            resume_text = uploaded_file.read().decode("utf-8")
        
        if resume_text.strip():
            # Display the extracted resume text
            st.subheader("Extracted Resume Text")
            st.text_area("Resume Content", value=resume_text, height=300)
            
            # Generate AI feedback
            st.subheader("AI Feedback")
            with st.spinner("Analyzing your resume..."):
                try:
                    feedback = resume_review_chain.run({"resume_text": resume_text})
                    st.write(feedback)
                except Exception as e:
                    st.error(f"Error generating feedback: {e}")
        else:
            st.error("Could not extract text from the uploaded file. Please try again.")

# Tab 2: HR Question Assistant
with tab2:
    st.header("HR Question Assistant")
    st.write("Ask any HR-related questions, and the AI will provide professional advice.")

    # Input box for HR question
    question = st.text_input("Enter your HR-related question:", value=selected_question)

    if question.strip():
        # Generate AI response
        st.subheader("AI Response")
        with st.spinner("Generating response..."):
            try:
                answer = hr_question_chain.run({"question": question})
                st.write(answer)
            except Exception as e:
                st.error(f"Error generating response: {e}")
