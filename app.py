import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain_groq import ChatGroq
import PyPDF2
import docx

# Define tools for resume review
def analyze_text_tool(text):
    """Tool to analyze resume text."""
    return f"Analyzing the following text: {text}"

def suggest_feedback_tool(text):
    """Tool to provide actionable feedback."""
    return f"Providing feedback for: {text}"

analyze_tool = Tool(
    name="AnalyzeText",
    func=analyze_text_tool,
    description="Analyzes the resume text for key insights and strengths.",
)

feedback_tool = Tool(
    name="SuggestFeedback",
    func=suggest_feedback_tool,
    description="Suggests actionable feedback for improvement.",
)

# Custom prompt template to enforce structure
CUSTOM_PROMPT_TEMPLATE = """
You are a professional resume assistant. Always follow the format below:

Thought: [Describe your reasoning]
Action: [Choose one of the tools: AnalyzeText or SuggestFeedback]
Action Input: [Provide the input text for the tool]

Observation: [Describe the result of the tool’s execution, only if an action is executed]

Final Answer: [Provide your complete and final response]

---

Input: {input}
"""

# Define agents
def get_resume_analyst(api_key):
    """Creates an agent for analyzing resumes."""
    llm = ChatGroq(
        model="gemma-7b-it",
        verbose=True,
        temperature=0.1,
        groq_api_key=api_key,
    )
    return initialize_agent(
        tools=[analyze_tool],
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True,
        prompt_template=CUSTOM_PROMPT_TEMPLATE,
    )

def get_feedback_specialist(api_key):
    """Creates an agent for providing actionable resume feedback."""
    llm = ChatGroq(
        model="gemma-7b-it",
        verbose=True,
        temperature=0.1,
        groq_api_key=api_key,
    )
    return initialize_agent(
        tools=[feedback_tool],
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True,
        prompt_template=CUSTOM_PROMPT_TEMPLATE,
    )

# Workflow for resume review
def create_resume_review_workflow(resume_text, api_key):
    """
    Executes the resume review workflow using Groq-powered agents.
    """
    resume_analyst = get_resume_analyst(api_key)
    feedback_specialist = get_feedback_specialist(api_key)

    try:
        analysis = resume_analyst.run(resume_text)
    except Exception as e:
        raise ValueError(f"Error in Resume Analyst: {e}")

    try:
        feedback = feedback_specialist.run(resume_text)
    except Exception as e:
        raise ValueError(f"Error in Feedback Specialist: {e}")

    return {
        "analysis": analysis,
        "feedback": feedback,
    }

# Functions for extracting text
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Streamlit App
st.title("Resume Review App")
st.write("Upload your resume to receive a detailed analysis and actionable feedback.")

# Sidebar for API key input
with st.sidebar:
    st.header("API Key Configuration")
    api_key = st.text_input(
        "Enter your GROQ_API_KEY:",
        type="password",
        placeholder="Enter your API key",
    )
    if api_key:
        st.success("✅ API Key Saved")
    else:
        st.warning("Please enter your API key to proceed.")

# Main interface
if api_key:
    uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

    if uploaded_file and st.button("Review Resume"):
        with st.spinner("Processing your resume..."):
            try:
                # Extract text from uploaded file
                if uploaded_file.type == "application/pdf":
                    resume_text = extract_text_from_pdf(uploaded_file)
                else:
                    resume_text = extract_text_from_docx(uploaded_file)

                # Execute the resume review workflow
                result = create_resume_review_workflow(resume_text, api_key)
                st.success("Resume review completed!")
                st.markdown("### Analysis")
                st.markdown(result["analysis"])
                st.markdown("### Feedback")
                st.markdown(result["feedback"])
            except Exception as e:
                st.error(f"An error occurred: {e}")