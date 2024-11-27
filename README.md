# HR Solutions GPT

A Streamlit-based application that leverages AI to provide HR solutions and resume analysis using the CrewAI framework and Groq's language models.

## 🚀 Features

### 1. HR Questions Module
- Interactive Q&A system for HR-related queries
- Dual-agent analysis system:
  - HR Problem Analyst for challenge evaluation
  - Solution Architect for detailed, actionable solutions
- Support for multiple language models:
  - Gemma 7B
  - Llama 3.2
  - Mixtral 8x7B

### 2. Resume Review System
- Automated resume analysis and feedback
- Supports both PDF and DOCX file formats
- Dual-agent review system:
  - Resume Analyst for comprehensive evaluation
  - Feedback Specialist for actionable improvements
- Detailed analysis of structure, content, and formatting

## 🛠️ Technical Requirements

```python
# Required Python packages
streamlit
crewai
langchain-groq
python-dotenv
PyPDF2
python-docx
```
## ⚙️ Setup
1. Clone the repository
   
2. Install dependencies:
- pip install -r requirements.txt
3. Create a .env file with your Groq API key:
- GROQ_API_KEY=your_api_key_here
streamlit run hrai.py
4. Run the Streamlit app:
- streamlit run hrai.py


## 🎯 Usage
### HR Questions
1. Select your preferred language model
2. Enter your HR-related question in the text area
3. Click "Get Answer" to receive detailed analysis and solutions

### Resume Review
1. Choose your preferred language model
2. Upload a resume in PDF or DOCX format
3. Click "Review Resume" to get comprehensive feedback

## 🤖 AI Agents
### HR Questions Module
- HR Problem Analyst: Analyzes challenges and provides contextual insights
- Solution Architect: Develops practical, implementable solutions

### Resume Review Module
- Resume Analyst: Evaluates resume structure and content
- Feedback Specialist: Provides actionable improvement suggestions

## 🔑 Key Features
- Real-time AI-powered analysis
- Support for multiple file formats
- Comprehensive feedback system
- User-friendly interface
- Example questions for reference

## 🔒 Privacy & Security
- Local file processing
- Secure API communication
- Environment variable protection for API keys
