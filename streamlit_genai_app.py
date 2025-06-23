from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import os
def get_file_contents(filename):
    """ Given a filename,
        return the contents of that file
    """
    try:
        with open(filename, 'r') as f:
            # It's assumed our file contains a single line,
            # with our API key
            return f.read().strip()
    except FileNotFoundError:
        print("'%s' file not found" % filename)

filename = "..\\GoogleAPIKey.txt"
os.environ['GOOGLE_API_KEY'] = get_file_contents(filename)
filename_grokKey = "..\\GroqAPIKey.txt"
os.environ['GROQ_API_KEY'] = get_file_contents(filename_grokKey)

llm = ChatGroq(model="llama-3.1-8b-instant")

search = DuckDuckGoSearchResults()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [search, wikipedia]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Based on user query and the chat history, look for information using DuckDuckGo Search and Wikipedia and then give the final answer",
        ),
        ("placeholder", "{history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


import streamlit as st

st.title("Ask me anything")


question = st.text_input("Question: ")

btn = st.button("Find Answer")

if btn:
    result = agent_with_history.invoke({"input": question}, config={"configurable": {"session_id": "sess1"}},)
    answer = result["output"]
    st.write(answer)



