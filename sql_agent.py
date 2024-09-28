import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd

from sqlalchemy import create_engine

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

llm_name = "gpt-3.5-turbo"
model = ChatOpenAI(api_key=openai_key, model=llm_name)

# df = pd.read_csv('./data/realtor-data.csv').fillna(value=0)

from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

database_file_path = "./db/listings.db"

engine = create_engine(f"sqlite:///{database_file_path}")
file_url = "./data/realtor-data.csv"
os.makedirs(os.path.dirname(database_file_path), exist_ok=True)
df = pd.read_csv(file_url).fillna(value=0)
df.to_sql("realtor_data", con=engine, if_exists="replace", index=False)

# print(f"Database created successfully! {df}")

MSSQL_AGENT_PREFIX = """

You are an agent designed to interact with a SQL database.
## Instructions:
- Given an input question, create a syntactically correct {dialect} query
to run, then look at the results of the query and return the answer.
- Unless the user specifies a specific number of examples they wish to
obtain, **ALWAYS** limit your query to at most {top_k} results.
- You can order the results by a relevant column to return the most
interesting examples in the database.
- Never query for all the columns from a specific table, only ask for
the relevant columns given the question.
- You have access to tools for interacting with the database.
- You MUST double check your query before executing it.If you get an error
while executing a query,rewrite the query and try again.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)
to the database.
- DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS
OF THE CALCULATIONS YOU HAVE DONE.
- Your response should be in Markdown. However, **when running  a SQL Query
in "Action Input", do not include the markdown backticks**.
Those are only for formatting the response, not for executing the command.
- ALWAYS, as part of your final answer, explain how you got to the answer
on a section that starts with: "Explanation:". Include the SQL query as
part of the explanation section.
- If the question does not seem related to the database, just return
"I don\'t know" as the answer.
- Only use the below tools. Only use the information returned by the
below tools to construct your query and final answer.
- Do not make up table names, only use the tables returned by any of the
tools below.
- as part of your final answer, please include the SQL query you used in json format or code format

## Tools:

"""

MSSQL_AGENT_FORMAT_INSTRUCTIONS = """

## Use the following format:

Question: the input question you must answer.
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: the final answer to the original input question.

Example of Final Answer:
<=== Beginning of example

Action: query_sql_db
Action Input:
SELECT COUNT(listing_id) FROM listings WHERE state = 'Washington';

Observation:
[(5000,)]  # Assuming this is the count returned by the query
Thought: I now know the final answer
Action: query_sql_db
Action Input:
SELECT COUNT(listing_id) FROM listings WHERE city = 'Seattle' AND state = 'Washington';

Observation:
[(1200,)]  # Assuming this is the count returned by the query
Thought: I now know the final answer

Final Answer: There are 5000 listings in Washington state and 1200 listings in Seattle, Washington.

Explanation:
I queried the `realtor_data` table to count the number of listings in Washington state and separately queried the number of listings specifically in Seattle, Washington. The queries returned counts of 5000 listings in Washington and 1200 in Seattle. To get these numbers, I used the following SQL queries:


===> End of Example

"""

db = SQLDatabase.from_uri(f"sqlite:///{database_file_path}")
toolkit = SQLDatabaseToolkit(db=db, llm=model)

QUESTION = """How many listings in Seattle, Washington have a 1 bed and 1 bath layout?"""

sql_agent = create_sql_agent(
    prefix = MSSQL_AGENT_PREFIX,
    format_instructions=MSSQL_AGENT_FORMAT_INSTRUCTIONS,
    llm=model,
    toolkit=toolkit,
    verbose=True,
)

res = sql_agent.invoke(QUESTION)
print(res)