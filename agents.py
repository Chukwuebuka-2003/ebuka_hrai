from crewai import Agent

def create_resume_analyst(llm):
    return Agent(
        llm=llm,
        role="Resume Analyst",
        goal="Analyze resumes and provide detailed feedback",
        backstory="You're an experienced HR professional specialized in resume screening "
                  "and providing constructive feedback to candidates.",
        allow_delegation=False,
        verbose=True
    )

def create_feedback_specialist(llm):
    return Agent(
        llm=llm,
        role="Feedback Specialist",
        goal="Provide actionable improvement suggestions",
        backstory="You're an expert in career development and resume optimization, "
                  "focusing on providing specific, actionable feedback.",
        allow_delegation=False,
        verbose=True
    )

def create_hr_problem_analyst(llm):
    return Agent(
        llm=llm,
        role="HR Problem Analyst",
        goal="Analyze HR challenges and provide practical solutions",
        backstory="You're an experienced HR consultant specialized in providing "
                  "actionable advice for HR-related questions and challenges.",
        allow_delegation=False,
        verbose=True
    )

def create_solution_architect(llm):
    return Agent(
        llm=llm,
        role="Solution Architect",
        goal="Provide detailed, implementable solutions",
        backstory="You're an HR solution expert who provides specific, "
                  "practical solutions with examples and best practices.",
        allow_delegation=False,
        verbose=True
    )
