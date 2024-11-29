from crewai import Task, Crew
from agents import (
    create_resume_analyst,
    create_feedback_specialist,
    create_hr_problem_analyst,
    create_solution_architect
)

def create_resume_review_crew(resume_text, llm):
    resume_analyst = create_resume_analyst(llm)
    feedback_specialist = create_feedback_specialist(llm)

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

def create_hr_crew(question, llm):
    hr_analyst = create_hr_problem_analyst(llm)
    solution_architect = create_solution_architect(llm)

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
