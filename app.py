import streamlit as st
from groq import Groq
import PyPDF2
import docx

# Configure page
st.set_page_config(page_title="HR Solutions Generator", page_icon="👥")

# Initialize the Groq client
@st.cache_resource
def initialize_llm(api_key):
    try:
        # Initialize the Groq client without any additional parameters
        return Groq(api_key=api_key)
    except TypeError as e:
        st.error(f"Failed to initialize Groq client: {str(e)}")
        return None

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

def chat_with_gemma(prompt, llm):
    try:
        response = llm.chat.completions.create(
            model="gemma-7b-it",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096  # Adjust according to your needs
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error communicating with the model: {str(e)}")
        return None

def create_resume_review(resume_text, llm):
    analysis_prompt = (
        f"Analyze this resume:\n{resume_text}\n"
        "1. Evaluate the overall structure and format\n"
        "2. Assess the content quality and relevance\n"
        "3. Identify strengths and weaknesses\n"
        "4. Check for essential components"
    )
    return chat_with_gemma(analysis_prompt, llm)

def create_hr_solution(question, llm):
    question_prompt = (
        f"Analyze this HR question or challenge: {question}\n"
        "1. Identify the core issue\n"
        "2. Consider relevant HR best practices\n"
        "3. Note any potential complications"
    )
    return chat_with_gemma(question_prompt, llm)

# Streamlit UI
st.title("HR Solutions GPT")
st.write("Ask HR questions or get resume feedback")

# API Key Input
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

api_key_input = st.sidebar.text_input("Enter your API key:", type="password")

# Store the API key in session state
if api_key_input:
    st.session_state.api_key = api_key_input

# Initialize the LLM if API key is provided
if st.session_state.api_key:
    llm = initialize_llm(st.session_state.api_key)

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
                    result = create_hr_solution(question, llm)
                    if result:
                        st.success("Response generated!")
                        st.markdown("### Solution")
                        st.markdown(result)
                    else:
                        st.error("Failed to generate a response.")
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

                    result = create_resume_review(resume_text, llm)
                    if result:
                        st.success("Resume review completed!")
                        st.markdown("### Resume Analysis")
                        st.markdown(result)
                    else:
                        st.error("Failed to analyze the resume.")
                except Exception as e:
                    st.error(f"An error occurred while processing the resume: {str(e)}")
else:
    st.warning("Please enter your API key to proceed.")