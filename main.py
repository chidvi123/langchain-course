from dotenv import load_dotenv

load_dotenv()
#uv add langchain-ollama langchain-tavily tavily-python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from tavily import TavilyClient
from langchain_tavily import TavilySearch

from langchain_ollama import ChatOllama

tavily=TavilyClient()

@tool
def search(query:str) -> str:
    """
    Tool that searches over internet
    Args :
        query : The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    response= tavily.search(query=query,max_results=3)
    results = response['results']

    return str(results)

'''
llm=ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)
'''

llm=ChatOllama(
    model='qwen3:8b',
    temperature=0
)

#tools=[search]

tools=[TavilySearch()]

agent = create_agent(model=llm,tools=tools)

def main():
    print("Hello from langchain-course!")
    result=agent.invoke(
        {'messages':HumanMessage(content=("Find 3 AI Engineer jobs in the Bay Area mentioning LangChain."))}
    )


    print(result)

if __name__=="__main__":
    main()