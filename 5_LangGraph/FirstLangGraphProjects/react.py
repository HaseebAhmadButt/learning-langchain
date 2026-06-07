# This file is going to be the reasoning engine for ReAct agent using LangGraph.

from dotenv import load_dotenv
from langchain_core.tools import tool # This is preferred because it is the base decorator

# from langchain.tools import tool # It is still the same as imported above.import

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_tavily import TavilySearch

load_dotenv()

@tool()
def triple(num: float) -> float:
    """
    :param num: a number to triple
    :return: the triple of the input number
    """
    return float(num) * 3





tools = [TavilySearch(max_results=1), triple]
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0).bind_tools(tools)