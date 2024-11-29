import streamlit as st
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
import PyPDF2
import docx
import io

# Configure page
st.set_page_config(page_title="HR Solutions Generator", page_icon="👥")

# Available LLM models
AVAILABLE_MODELS = {
    "Gemma 7B": "gemma-7b-it",
    "Llama 3.2": "llama-3.2-3b-preview",
    "Mixtral 8x7B": "mixtral-8x7b-32768"
}

@st.cache_resource
def initialize_llm(model_name, api_key):
    return ChatGroq(
        temperature=0,
        model_name=model_name,
        api_key=api_key
    )

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

def create_resume_review_crew(resume_text, model_name, api_key):
    llm = initialize_llm(model_name, api_key)
    resume_analyst = Agent(
        llm=llm,
        role="Resume Analyst",
        goal="Analyze resumes and provide detailed feedback",
        backstory="You're an experienced HR professional specialized in resume screening "
                 "and providing constructive feedback to candidates.",
        allow_delegation=False,
        verbose=True
    )
    feedback_specialist = Agent(
        llm=llm,
        role="Feedback Specialist",
        goal="Provide actionable improvement suggestions",
        backstory="You're an expert in career development and resume optimization, "
                 "focusing on providing specific, actionable feedback.",
        allow_delegation=False,
        verbose=True
    )
    analyze_resume = Task(
        description=(
            f"Analyze this resume:\n{resume_text}\n"
            "1. Evaluate the overall structure and format\n"
            "2. Assess the content quality and relevance\n"
            "3. Identify strengths and weaknesses\n"
            "4. Check for essential components"
        ),
        expected_output="Detailed resume analysis",
        agent=resume_analyst
    )
    provide_feedback = Task(
        description=(
            "Based on the analysis, provide detailed feedback:\n"
            "1. List specific improvements needed\n"
            "2. Highlight positive aspects\n"
            "3. Suggest concrete changes\n"
            "4. Provide formatting recommendations"
        ),
        expected_output="Comprehensive feedback with actionable suggestions",
        agent=feedback_specialist
    )
    return Crew(
        agents=[resume_analyst, feedback_specialist],
        tasks=[analyze_resume, provide_feedback],
        verbose=2
    )

def create_hr_crew(question, model_name, api_key):
    llm = initialize_llm(model_name, api_key)
    hr_analyst = Agent(
        llm=llm,
        role="HR Problem Analyst",
        goal="Analyze HR challenges and provide practical solutions",
        backstory="You're an experienced HR consultant specialized in providing "
                 "actionable advice for HR-related questions and challenges.",
        allow_delegation=False,
        verbose=True
    )
    solution_architect = Agent(
        llm=llm,
        role="Solution Architect",
        goal="Provide detailed, implementable solutions",
        backstory="You're an HR solution expert who provides specific, "
                 "practical solutions with examples and best practices.",
        allow_delegation=False,
        verbose=True
    )
    analyze_question = Task(
        description=(
            f"Analyze this HR question or challenge: {question}\n"
            "1. Identify the core issue\n"
            "2. Consider relevant HR best practices\n"
            "3. Note any potential complications"
        ),
        expected_output="Clear analysis with key considerations",
        agent=hr_analyst
    )
    provide_solution = Task(
        description=(
            f"Based on the analysis, address this question: {question}\n"
            "1. Provide specific, actionable solutions\n"
            "2. Include relevant examples\n"
            "3. Add implementation tips\n"
            "4. Consider different organizational contexts"
        ),
        expected_output="Detailed, practical solution with examples",
        agent=solution_architect
    )
    return Crew(
        agents=[hr_analyst, solution_architect],
        tasks=[analyze_question, provide_solution],
        verbose=2
    )

# Streamlit UI
st.title("HR Solutions GPT")
st.write("Ask HR questions or get resume feedback")

# API Key Input
api_key = st.sidebar.text_input("Enter your API key:", type="password")

# Sidebar with examples
with st.sidebar:
    st.header("Example Questions")
    examples = [
        "How can I improve employee retention in a remote work environment?",
        "What are effective strategies for conducting performance reviews?",
        "How should I handle conflicts between team members?",
        "What's the best way to implement a new training program?",
        "How can I create an effective employee onboarding process?"
    ]
    st.write("Try asking questions like:")
    for example in examples:
        st.write(f"• {example}")

if not api_key:
    st.warning("Please enter your API key in the sidebar to proceed.")
else:
    # Tab selection
    tab1, tab2 = st.tabs(["HR Questions", "Resume Review"])

    with tab1:
        selected_model = st.selectbox(
            "Select Language Model:",
            options=list(AVAILABLE_MODELS.keys()),
            key="hr_model"
        )
        question = st.text_area(
            "Enter your HR-related question:",
            height=100,
            placeholder="e.g., How can I improve team collaboration in a hybrid workplace?"
        )
        if st.button("Get Answer"):
            if question:
                with st.spinner("Analyzing and generating response..."):
                    try:
                        model_name = AVAILABLE_MODELS[selected_model]
                        crew = create_hr_crew(question, model_name, api_key)
                        result = crew.kickoff(inputs={"question": question})
                        st.success("Response generated!")
                        st.markdown("### Solution")
                        st.markdown(result)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("Please enter a question.")

    with tab2:
        selected_model = st.selectbox(
            "Select Language Model:",
            options=list(AVAILABLE_MODELS.keys()),
            key="resume_model"
        )
        uploaded_file = st.file_uploader(
            "Upload your resume (PDF or DOCX)",
            type=["pdf", "docx"]
        )
        if uploaded_file and st.button("Review Resume"):
            with st.spinner("Analyzing resume..."):
                try:
                    if uploaded_file.type == "application/pdf":
                        resume_text = extract_text_from_pdf(uploaded_file)
                    else:
                        resume_text = extract_text_from_docx(uploaded_file)

                    model_name = AVAILABLE_MODELS[selected_model]
                    crew = create_resume_review_crew(resume_text, model_name, api_key)
                    result = crew.kickoff(inputs={"resume_text": resume_text})
                    st.success("Resume review completed!")
                    st.markdown("### Resume Analysis")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
