from flask import Flask, request, jsonify
from crewai import Agent, Task, Crew
from litellm import completion  # Import LiteLLM completion
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
    """
    Initializes the connection to the Groq model using LiteLLM.
    """
    prefixed_model_name = f"groq/{model_name}"  # Add Groq prefix
    return prefixed_model_name  # Return the fully qualified model name for use in LiteLLM

def analyze_with_llm(model_name, user_input):
    """
    Send user input to the Groq model using LiteLLM's completion API.
    Extracts and returns the model's content as plain text.
    """
    try:
        response = completion(
            model=model_name,
            messages=[
                {"role": "user", "content": user_input}
            ],
            api_key=os.getenv("GROQ_API_KEY")
        )
        # Extract the relevant content from the response
        if hasattr(response, "choices") and response.choices:
            return response.choices[0].message["content"]  # Adjust based on LiteLLM response structure
        else:
            raise ValueError("Unexpected response structure from the LLM")
    except Exception as e:
        raise ValueError(f"Error while communicating with the LLM: {str(e)}")

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

@app.route('/hr', methods=['POST'])
def analyze_hr_question():
    """
    Endpoint to analyze HR-related questions using Groq models.
    """
    data = request.json
    question = data.get('question')
    model_name = data.get('model_name')
    if not question or not model_name:
        return jsonify({"error": "Question and model name are required"}), 400
    if model_name not in AVAILABLE_MODELS.values():
        return jsonify({"error": "Invalid model name"}), 400

    try:
        full_model_name = initialize_llm(model_name)
        response_content = analyze_with_llm(full_model_name, question)
        return jsonify({"result": response_content}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/resume', methods=['POST'])
def review_resume():
    """
    Endpoint to analyze resumes using Groq models.
    """
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

        full_model_name = initialize_llm(model_name)
        response_content = analyze_with_llm(full_model_name, resume_text)
        return jsonify({"result": response_content}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
