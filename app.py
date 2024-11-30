import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor
from langchain.prompts import PromptTemplate

# Set Streamlit Page Config
st.set_page_config(
    page_title="AI-Powered Career Tools",
    page_icon="🤖",
    layout="wide"
)

# Initialize the Google Gemini LLM
api_key = st.secrets["google_genai"]["api_key"]  # Replace with your actual API key
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)

# Define the prompt template for resume review
resume_prompt_template = PromptTemplate(
    input_variables=["resume_text"],
    template=(
        "You are an experienced resume reviewer. Analyze the following resume and provide constructive feedback:\n\n"
        "{resume_text}\n\n"
        "Include feedback on strengths, weaknesses, skills, formatting, and language improvement."
    ),
)

# Define the prompt template for career questions
career_prompt_template = PromptTemplate(
    input_variables=["query"],
    template=(
        "You are a career coach. Provide a detailed and professional answer to the following question:\n\n"
        "{query}"
    ),
)

# Streamlit App with Tabs
st.title("AI-Powered Career Tools")
tab1, tab2 = st.tabs(["Resume Reviewer", "Career Guidance Assistant"])

# Agent Executor and Tool Setup
tools = []

# Tab 1: Resume Reviewer
with tab1:
    st.header("Resume Reviewer")
    st.write("Upload your resume to receive actionable feedback.")

    # File uploader for resumes
    uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])

    if uploaded_file:
        # Extract text from the uploaded file
        resume_text = ""
        if uploaded_file.type == "application/pdf":
            try:
                reader = PdfReader(uploaded_file)
                resume_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
        elif uploaded_file.type == "text/plain":
            resume_text = uploaded_file.read().decode("utf-8")

        if resume_text.strip():
            # Display the extracted resume text
            st.subheader("Extracted Resume Text")
            st.text_area("Resume Content", value=resume_text, height=300)

            # Add Resume Reviewer Tool
            tools.append(
                Tool(
                    name="ResumeReviewer",
                    description="Provides feedback on resumes.",
                    func=lambda input_text: llm.generate_response(
                        resume_prompt_template.format(resume_text=input_text)
                    ),
                )
            )

# Tab 2: Career Guidance Assistant
with tab2:
    st.header("Career Guidance Assistant")
    st.write("Ask career-related questions and receive expert advice.")

    # Text input for general career questions
    query = st.text_area("Ask your question here:")

    if query.strip():
        # Add Career Guidance Tool
        tools.append(
            Tool(
                name="CareerAssistant",
                description="Provides professional career advice.",
                func=lambda input_text: llm.generate_response(
                    career_prompt_template.format(query=input_text)
                ),
            )
        )

# Agent Executor
if tools:
    agent = AgentExecutor(
        tools=tools,
        llm=llm,  # Use the initialized Google GenAI model
    )

    # Run Agent Based on Input
    if resume_text.strip() and uploaded_file:
        st.subheader("AI Feedback for Resume")
        with st.spinner("Analyzing your resume..."):
            try:
                resume_feedback = agent.run(resume_text)
                st.write(resume_feedback)
            except Exception as e:
                st.error(f"Error generating feedback: {e}")

    if query.strip():
        st.subheader("AI Response to Career Query")
        with st.spinner("Processing your question..."):
            try:
                career_feedback = agent.run(query)
                st.write(career_feedback)
            except Exception as e:
                st.error(f"Error generating response: {e}")
