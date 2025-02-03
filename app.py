__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from tenacity import retry, wait_exponential, stop_after_attempt

# Streamlit UI
st.set_page_config(layout="wide")
st.sidebar.title("About")
st.sidebar.write("This app leverages CrewAI to research, write, and edit AI-generated blog content based on a user-provided topic.")
st.title("AI-Powered Blog Content Generator")


# Set API Keys
st.sidebar.title("API Key Configuration")
os.environ["SERPER_API_KEY"] = st.sidebar.text_input("Enter Serper API Key:", type="password")
os.environ['OPENAI_API_KEY'] = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
os.environ["OPENAI_MODEL_NAME"] = st.sidebar.selectbox("Select OpenAI Model:", ["gpt-4o-mini-2024-07-18", "gpt-4", "gpt-3.5-turbo"])
# User Inputs
topic = st.text_input("Enter a Topic")

# Set API Keys
if os.environ["SERPER_API_KEY"] and os.environ['OPENAI_API_KEY']:

    # Tools
    SerperDevTool = SerperDevTool()
    ScrapeWebsiteTool = ScrapeWebsiteTool()

    # Agents
    planner = Agent(
        role="Research & Content Planner",
        goal=(
            "Conduct thorough research and curate factually accurate, engaging content on {topic}. "
            "Ensure the collected information is credible, up-to-date, and valuable for the target audience."
        ),
        backstory=(
            "A meticulous Content Planner with expertise in research, content strategy, and audience engagement. "
            "With a keen eye for detail and a strong analytical mindset, they specialize in gathering reliable data, "
            "identifying key trends, and structuring insights into well-organized content plans. "
            "Their experience spans digital content creation, market research, and fact-checking, ensuring "
            "that every piece of information is credible and impactful. "
            "They excel at distilling complex topics into clear, actionable insights that serve as a foundation "
            "for compelling content creation."
        ),
        allow_delegation=False,
        verbose=True
    )

    writer = Agent(
        role="Content Writer",
        goal="Write insightful and factually accurate "
             "opinion piece about the topic: {topic}",
        backstory="You're working on a writing "
                  "a new opinion piece about the topic: {topic}. "
                  "You base your writing on the work of "
                  "the Research & Content Planner, who provides an outline "
                  "and relevant context about the topic. "
                  "You follow the main objectives and "
                  "direction of the outline, "
                  "as provide by the Content Planner. "
                  "You also provide objective and impartial insights "
                  "and back them up with information "
                  "provide by the Content Planner. "
                  "You acknowledge in your opinion piece "
                  "when your statements are opinions "
                  "as opposed to objective statements.",
        allow_delegation=False,
        verbose=True
    )

    editor = Agent(
        role="Editor",
        goal="Edit a given blog post to align with "
             "the writing style of the organization. ",
        backstory="You are an editor who receives a blog post "
                  "from the Content Writer. "
                  "Your goal is to review the blog post "
                  "to ensure that it follows journalistic best practices,"
                  "provides balanced viewpoints "
                  "when providing opinions or assertions, "
                  "and also avoids major controversial topics "
                  "or opinions when possible.",
        allow_delegation=False,
        verbose=True
    )

    # Tasks
    plan = Task(
        description=(
            "1. Prioritize the latest trends, key players, "
                "and noteworthy news on {topic}.\n"
            "2. Identify the target audience, considering "
                "their interests and pain points.\n"
            "3. Develop a detailed content outline including "
                "an introduction, key points, and a call to action, conclusion and necessary referernces.\n"
            "4. Include SEO keywords and relevant data or sources."
        ),
        expected_output="A comprehensive content plan document "
            "with an outline, audience analysis, "
            "SEO keywords, and resources.",
        tools=[SerperDevTool, ScrapeWebsiteTool],
        agent=planner,
    )
    
    write = Task(
        description=(
            "1. Use the content plan to craft a compelling "
                "blog post on {topic}.\n"
            "2. Incorporate SEO keywords naturally.\n"
                "3. Sections/Subtitles are properly named "
                "in an engaging manner.\n"
            "4. Adding necessary Hyperlinks and bolding for important sentances/words/statements.\n"
            "5. If required you can add comparision tables/table and data if it is necessary for the topic.\n"
            "6. Ensure the post is structured with an "
                "engaging introduction, insightful body, "
                "and a summarizing conclusion.\n"
            "7. Proofread for grammatical errors and "
                "alignment with the brand's voice.\n"
        ),
        expected_output="A well-written blog post "
            "in markdown format, ready for publication, "
            "each section should have 2 or 3 paragraphs.",
        tools=[SerperDevTool, ScrapeWebsiteTool],
        agent=writer,
    )
    
    edit = Task(
        description=("Proofread the given blog post for "
                     "grammatical errors, checks plagiarism and "
                     "alignment with the brand's voice."),
        expected_output="A well-written blog post, "
                        "ready for publication, "
                        "each section should have 2 or 3 paragraphs.",
        tools=[SerperDevTool, ScrapeWebsiteTool],
        agent=editor,
        output_file="Final_Article.txt",
    )

    # Crew
    crew = Crew(
        agents=[planner, writer, editor],
        tasks=[plan, write, edit],
        verbose=True
    )

    @retry(wait=wait_exponential(multiplier=1, max=60), stop=stop_after_attempt(5))
    def kickoff_with_retry(crew, inputs):
        return crew.kickoff(inputs)

import streamlit as st

if st.button("Generate Article"):
    if topic:
        try:
            result = kickoff_with_retry(crew, inputs={"topic": topic})
            if result:

                # Read and display the article from the file
                with open("Final_Article.txt", "r") as file:
                    Final_Article = file.read()
                    cleaned_output = str(Final_Article).strip().replace("markdown ```", "").strip()

                st.subheader("Final Article")
                st.markdown(cleaned_output)

                # Download button
                st.download_button(label="Download Final Article", data=cleaned_output, 
                                   file_name="Final_Article.txt", mime="text/plain")
            else:
                st.error("Failed to generate the article. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a topic before generating.")

