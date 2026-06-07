from dotenv import load_dotenv
from langchain_core.tools import tool # This is preferred because it is the base decorator

# from langchain.tools import tool # It is still the same as imported above.import

from langchain_openai import OpenAI
from langchain_tavily import TavilySearch