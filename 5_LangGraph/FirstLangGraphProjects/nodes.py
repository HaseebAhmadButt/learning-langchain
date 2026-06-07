from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
# This is going to be a list of messages.
# This is going to keep track of the state on all
# the messages back in between our agents, i.e., Human Message and AI Message.
from langgraph.graph import MessagesState

# This is going to execute the tool based on the last state.
# For example, if it is the AI Message, then it is going to execute the tool and update the state with the tool's output.
# Assuming that tool is initiated with the tool node object.
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

SYSTEM_MESSAGE = """
You are a helpful assistant that use tools to answer questions.
"""

# This is the reasoning node
def run_agent_reasoning(state: MessagesState)-> MessagesState:
    """
    Run the agent reasoning node.
    """
    response = llm.invoke([
        SystemMessage(content=SYSTEM_MESSAGE),
        *state["messages"]
    ])
    return {"messages": [response]}

tool_node = ToolNode(tools)