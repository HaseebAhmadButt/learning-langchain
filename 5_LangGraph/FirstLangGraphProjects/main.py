from dotenv import load_dotenv
from langgraph.constants import START, END

load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph

from nodes import run_agent_reasoning, tool_node

AGENT_REASONING="agent_reason"
ACT="act"
LAST=-1

def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT


# Create a state-based graph where message history is stored and passed between nodes
flow = StateGraph(MessagesState)

# Add a reasoning node that runs the LLM agent logic
flow.add_node(AGENT_REASONING, run_agent_reasoning)

# Set the entry point of the graph to the reasoning node
flow.set_entry_point(AGENT_REASONING)

# Add a tool execution node used when the agent decides to take an action
flow.add_node(ACT, tool_node)

# Define the starting transition into the reasoning node
flow.add_edge(START, AGENT_REASONING)



# add_conditional_edges() creates dynamic routing after a node executes.
# After AGENT_REASONING completes, LangGraph calls should_continue()
# to decide the next destination.
#
# If should_continue() returns ACT:
#     AGENT_REASONING -> ACT
#
# If should_continue() returns END:
#     AGENT_REASONING -> END
#
# Routing map:
# {
#     END: END,
#     ACT: ACT
# }
#
# Means:
# - Return value END routes to END node.
# - Return value ACT routes to ACT node.
#
# Commonly used in agent loops where the agent decides whether
# to invoke a tool (ACT) or finish execution (END).
flow.add_conditional_edges(
    AGENT_REASONING,
    should_continue,
    {
        END: END,
        ACT: ACT
    }
)

# Normally, you can define a fixed path:
# flow.add_edge("reasoning", "act")
# Which means:
# reasoning --> act
# Every time the "reasoning" node finishes, execution always moves to "act".

flow.add_edge(ACT, AGENT_REASONING)

app = flow.compile()
app.get_graph().draw_png(output_file_path="agent_loop.png")
if __name__ == '__main__':
    print("Hello ReAct LangGraph with Function Calling")
    res = app.invoke({"messages": [HumanMessage(content="What is the weather in Lahore? List it and then triple it")]})
    print(res["messages"][LAST].content)