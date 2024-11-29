import streamlit as st
from langchain_groq import ChatGroq
from tasks import create_resume_review_crew, create_hr_crew
import PyPDF2
import docx

# Configure page
st.set_page_config(page_title="HR Solutions Generator", page_icon="👥")

# Default model
MODEL_NAME = "gemma-7b-it"

@st.cache_resource
def initialize_llm(api_key):
    return ChatGroq(
        temperature=0,
        model_name=MODEL_NAME,
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

# Initialize session state for API key
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""

def save_api_key():
    """Save the user-provided API key to session state."""
    st.session_state["api_key"] = st.session_state["input_api_key"]
    st.success("API key saved successfully!")

# Streamlit UI
st.title("HR Solutions GPT")
st.write("Ask HR questions or get resume feedback")

# Sidebar with API key input
with st.sidebar:
    st.header("Settings")
    st.text_input(
        "Enter your API key:",
        type="password",
        key="input_api_key",
        placeholder="Enter your API key",
    )
    st.button("Save API Key", on_click=save_api_key)

# Display saved API key status
if st.session_state["api_key"]:
    st.sidebar.write("✅ API Key Saved")

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

if not st.session_state["api_key"]:
    st.warning("Please enter and save your API key in the sidebar to proceed.")
else:
    # Tab selection
    tab1, tab2 = st.tabs(["HR Questions", "Resume Review"])

    with tab1:
        question = st.text_area(
            "Enter your HR-related question:",
            height=100,
            placeholder="e.g., How can I improve team collaboration in a hybrid workplace?"
        )
        if st.button("Get Answer"):
            if question:
                with st.spinner("Analyzing and generating response..."):
                    try:
                        llm = initialize_llm(st.session_state["api_key"])
                        crew = create_hr_crew(question, llm)
                        result = crew.kickoff(inputs={"question": question})
                        st.success("Response generated!")
                        st.markdown("### Solution")
                        st.markdown(result)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("Please enter a question.")

    with tab2:
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

                    llm = initialize_llm(st.session_state["api_key"])
                    crew = create_resume_review_crew(resume_text, llm)
                    result = crew.kickoff(inputs={"resume_text": resume_text})
                    st.success("Resume review completed!")
                    st.markdown("### Resume Analysis")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
