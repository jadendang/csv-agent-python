from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd

# to load the venv, run: .\venv/Scripts/Activate.ps1

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

llm_name = "gpt-3.5-turbo"
model = ChatOpenAI(api_key=openai_key, model=llm_name)

df = pd.read_csv('./data/realtor-data.csv').fillna(value=0)

# print(df.head())

from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent,
    create_csv_agent,
)

agent = create_pandas_dataframe_agent(
    llm = model,
    df = df,
    verbose = True,
)

# res = agent.invoke("how many listings are in washington state?")

# print(res)

CSV_PROMPT_PREFIX = """
First set the pandas display options to show all the columns,
get the column names, then answer the question.
"""

CSV_PROMPT_SUFFIX = """
- **ALWAYS** before giving the Final Answer, try another method.
Then reflect on the answers of the two methods you did and ask yourself
if it answers correctly the original question.
If you are not sure, try another method.
FORMAT 4 FIGURES OR MORE WITH COMMAS.
- If the methods tried do not give the same result,reflect and
try again until you have two methods that have the same result.
- If you still cannot arrive to a consistent result, say that
you are not sure of the answer.
- If you are sure of the correct answer, create a beautiful
and thorough response using Markdown.
- **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
- **ALWAYS**, as part of your "Final Answer", explain how you got
to the answer on a section that starts with: "\n\nExplanation:\n".
In the explanation, mention the column names that you used to get
to the final answer.
"""
QUESTION = "How many listings in Seattle, Washington have a 1 bed and 1 bath layout?"

res = agent.invoke(CSV_PROMPT_PREFIX + QUESTION + CSV_PROMPT_SUFFIX)

# print(f"Final result: {res["output"]}")

import streamlit as st

st.title("Real Estate AI Agent with LangChain")

st.write("### Dataset Preview")
st.write(df.head(10))

st.write("### Ask a Question")
question = st.text_input(
    "Enter your question about the dataset:",
    "How many listings in Seattle, Washington have a 1 bed and 1 bath layout?",
)

if st.button("Run Query"):
    QUERY = CSV_PROMPT_PREFIX + question + CSV_PROMPT_SUFFIX
    res = agent.invoke(QUERY)
    st.write("### Final Answer")
    st.write(res["output"])