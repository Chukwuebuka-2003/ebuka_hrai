import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor, tool

# Retrieve the API key from Streamlit secrets
api_key = st.secrets["google_genai"]["api_key"]

# Initialize the Google Gemini model
llm = ChatGoogleGenerativeAI(model="gemini1.5flash", temperature=0.3, google_api_key=api_key)

# Define a tool for resume review
@tool
def resume_review_tool(resume_text: str) -> str:
    """Provide detailed feedback on the given resume text."""
    prompt = (
        "You are a professional resume reviewer. Provide detailed feedback on the following resume:\n\n"
        f"{resume_text}\n\n"
        "Your feedback should include suggestions on structure, language, skills, and overall presentation."
    )
    response = llm.generate([prompt])
    return response.generations[0][0].text

# Create the agent using the tool
tools = [resume_review_tool]
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_tool_calling_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit UI
st.title("AI-Powered Resume Reviewer")
st.write("Upload your resume (PDF or TXT), and the AI will provide detailed feedback.")

uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "txt"])

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
        st.subheader("Uploaded Resume Text")
        st.text_area("Resume Content", value=resume_text, height=300)

        st.subheader("AI Feedback")
        with st.spinner("Analyzing your resume..."):
            try:
                feedback = agent_executor.invoke({"input": resume_text})
                st.write(feedback['output'])
            except Exception as e:
                st.error(f"Error generating feedback: {e}")
    else:
        st.error("Could not extract text from the uploaded file. Please try again.")
