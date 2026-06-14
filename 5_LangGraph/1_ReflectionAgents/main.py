from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generation_chain, reflection_chain

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph):
    return {"messages": [generation_chain.invoke({"messages": state["messages"]})]}

def reflection_node(state: MessageGraph):
    resp = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=resp.content)]}

def should_continue(state: MessageGraph) -> str:
    if len(state["messages"]) > 6: #Just a random number to stop the loop
        return END
    return REFLECT

# builder keyword is replaced with graph
graph = StateGraph(state_schema=MessageGraph)
graph.add_node(GENERATE, generation_node)
graph.add_node(REFLECT, reflection_node)
graph.set_entry_point(GENERATE)
graph.add_conditional_edges(GENERATE, should_continue,
# We can just remove this dictionary, and it will work fine.
# Because the nodes have the same name, it will automatically choose the correct node.
# If we don't add this dictionary, it will not be printed out when we generate the mermaid code of the graph or generate the image of the graph.
{
    END:END,
    REFLECT:REFLECT
}
)
graph.add_edge(REFLECT, GENERATE)
app = graph.compile()
print(app.get_graph().draw_mermaid())
# ---
# config:
#   flowchart:
#     curve: linear
# ---
# graph TD;
# 	__start__(<p>__start__</p>)
# 	generate(generate)
# 	reflect(reflect)
# 	__end__(<p>__end__</p>)
# 	__start__ --> generate;
# 	generate --> __end__;
# 	classDef default fill:#f2f0ff,line-height:1.2
# 	classDef first fill-opacity:0
# 	classDef last fill:#bfb6fc

if __name__ == "__main__":
    print("Hello, LangGraph Reflection Agents!")
    inputs = HumanMessage(content="""
    Make this tweet better:
    @LangChainAPI
    - newly Tool Calling feature is seriously underrated.
    After a long wait, it's here- making the implementation of agents across different models with function calling super easy.
    Made a video covering their newest blog post
    """)
    app.invoke(inputs)