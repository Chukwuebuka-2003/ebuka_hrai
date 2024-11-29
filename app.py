import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain_google_genai import ChatGoogleGenerativeAI  # Correct import for Gemini LLM

# Define tools for the conversational bot
def hr_tips_tool(question):
    """Tool to provide HR-related tips."""
    return f"Tips for your question: {question}"

hr_tool = Tool(
    name="HRTips",
    func=hr_tips_tool,
    description="Provides HR-related tips and suggestions.",
)

# Custom prompt template for the HR bot
CUSTOM_PROMPT_TEMPLATE = """
You are an HR expert assistant. Always follow this format:

Thought: [Explain your reasoning or thoughts]
Action: [Choose the tool: HRTips]
Action Input: [Provide the user's question or input]

Observation: [Describe the result from the tool’s execution]

Final Answer: [Provide a complete and concise response to the user's question]

---

Input: {input}
"""

# Define the HR Tips Conversational Bot Agent
def get_hr_bot_agent(api_key):
    """Creates an agent for HR tips and advice."""
    llm = ChatGoogleGenerativeAI(temperature=0.1, model="gemini-1", api_key=api_key)  # Pass user-provided API key
    return initialize_agent(
        tools=[hr_tool],
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True,
        prompt_template=CUSTOM_PROMPT_TEMPLATE,
    )

# Streamlit App
st.title("HR Tips Bot")
st.write("Ask HR-related questions and receive expert advice.")

# Sidebar for API key input
with st.sidebar:
    st.header("API Key Configuration")
    api_key = st.text_input(
        "Enter your Gemini API Key:",
        type="password",
        placeholder="Enter your API key here",
    )
    if api_key:
        st.success("✅ API Key Saved")
    else:
        st.warning("Please enter your Gemini API key to proceed.")

# Main interface
if api_key:
    user_question = st.text_area(
        "Enter your HR-related question:",
        placeholder="e.g., How do I improve employee retention?",
    )

    if st.button("Get HR Tips"):
        with st.spinner("Thinking..."):
            try:
                # Get HR Bot Agent with API key
                hr_bot = get_hr_bot_agent(api_key)

                # Generate a response
                response = hr_bot.run(user_question)
                st.success("Here's what I found:")
                st.markdown(f"**HR Advice:** {response}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.error("Please enter your Gemini API key in the sidebar to use the bot.")