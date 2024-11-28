from flask import Flask, request, jsonify
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import PyPDF2
import docx
import io

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Available LLM models
AVAILABLE_MODELS = {
    "Gemma 7B": "gemma-7b-it",
    "Llama 3.2": "llama3-8b-8192",
    "Mixtral 8x7B": "mixtral-8x7b-32768"
}

def initialize_llm(model_name):
    prefixed_model_name = f"groq/{model_name}"  # Add Groq prefix
    return ChatGroq(
        temperature=0,
        model_name=prefixed_model_name,
        model_kwargs={"provider": "groq"},  # Use model_kwargs for provider
        api_key=os.getenv("GROQ_API_KEY")
    )


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF files."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    """Extract text from DOCX files."""
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def create_resume_review_crew(resume_text, model_name):
    """Create a crew to review resumes."""
    llm = initialize_llm(model_name)

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
        verbose=True
    )

def create_hr_crew(question, model_name):
    """Create a crew to analyze HR questions."""
    llm = initialize_llm(model_name)

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
        verbose=True
    )

@app.route('/hr', methods=['POST'])
def analyze_hr_question():
    """Endpoint to analyze HR questions."""
    data = request.json
    question = data.get('question')
    model_name = data.get('model_name')
    if not question or not model_name:
        return jsonify({"error": "Question and model name are required"}), 400
    if model_name not in AVAILABLE_MODELS.values():
        return jsonify({"error": "Invalid model name"}), 400

    try:
        crew = create_hr_crew(question, model_name)
        result = crew.kickoff(inputs={"question": question})
        return jsonify({"result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/resume', methods=['POST'])
def review_resume():
    """Endpoint to review resumes."""
    if 'file' not in request.files or 'model_name' not in request.form:
        return jsonify({"error": "Resume file and model name are required"}), 400

    file = request.files['file']
    model_name = request.form['model_name']
    if model_name not in AVAILABLE_MODELS.values():
        return jsonify({"error": "Invalid model name"}), 400

    try:
        if file.content_type == 'application/pdf':
            resume_text = extract_text_from_pdf(file)
        elif file.content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            resume_text = extract_text_from_docx(file)
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        crew = create_resume_review_crew(resume_text, model_name)
        result = crew.kickoff(inputs={"resume_text": resume_text})
        return jsonify({"result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
