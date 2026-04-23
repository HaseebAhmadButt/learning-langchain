from dotenv import load_dotenv  # Import the load_dotenv function to load environment variables from a .env file
load_dotenv()  # Execute load_dotenv to load the environment variables

# This is the LangChain provided class to inegrate LLM models, independent of the LLM. Just change the name of the LLM and the parameters, and you can use any LLM you want.
from langchain.chat_models import init_chat_model  # Import init_chat_model for initializing chat models # https://reference.langchain.com/python/langchain/chat_models/base/init_chat_model
from langchain_ollama import ChatOllama  # Import ChatOllama for Ollama integration
from langchain.messages import HumanMessage, ToolMessage, SystemMessage  # Import message classes for chat
from langchain.tools import tool  # Import the tool decorator for defining tools
from langsmith import traceable  # Import traceable for tracing

MAX_ITERATIONS = 10  # Define the maximum number of iterations for the agent loop
MODEL_NAME = "qwen3-coder:30b"  # Define the model name for the LLM

@tool  # Decorator to mark this function as a tool
def get_product_price(product: str) -> str:  # Define a tool function to get product price
    """Look up the price of a product in the catalog"""  # Docstring for the function
    print(f"  >> Executing get_product_price(product='{product}')")  # Print execution message
    prices: dict[str, int] = {  # Define a dictionary of product prices
        "laptop": 999,  # Price for laptop
        "smartphone": 499,  # Price for smartphone
        "headphones": 199  # Price for headphones
    }
    return prices.get(product.lower(), 0)  # Return the price or 0 if not found

@tool  # Decorator to mark this function as a tool
def apply_discount(price: float, discount_tier: str) -> float:  # Define a tool function to apply discount
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold"""  # Docstring for the function
    print(f"  >> Executing apply_discount(price={price}, discount={discount_tier})")  # Print execution message
    discount_percentage: dict[str, float] = {  # Define a dictionary of discount percentages
       "bronze": 5,  # Bronze discount percentage
       "silver": 12,  # Silver discount percentage
       "gold": 23  # Gold discount percentage
    }
    discount = discount_percentage.get(discount_tier, 0)  # Get the discount percentage or 0
    return round(price * (1 - discount/100),2)  # Calculate and return the discounted price rounded to 2 decimals


#   ------- Agent Loop -------

@traceable(name="Langchain Agent Loop")  # Decorator for tracing the function
def run_agent(question: str):  # Define the main agent function
    tools = [get_product_price, apply_discount]  # List of available tools
    tools_dict = {t.name: t for t in tools}  # Create a dictionary of tools by name
    llm = init_chat_model(model=f'ollama:{MODEL_NAME}', temperature=0)  # Initialize the LLM model
    llm_with_tools =  llm.bind_tools(tools)  # Bind the tools to the LLM
    print(f"Question: '{question}')")  # Print the user's question
    print("=" * 60)  # Print a separator line

    messages = [  # Initialize the messages list
        SystemMessage(  # Create a system message
            content=(  # Content of the system message
                "You are a helpful shopping assistant. "  # Part of the content
                "You have access to a product catalog tool "  # Continuation
                "and a discount tool.\n\n"  # Continuation
                "STRICT RULES - you must follow these exactly:\n"  # Rules start
                "1. NEVER guess or assume any product price. "  # Rule 1
                "You MUST call get_product_price first to get the real price.\n"  # Continuation
                "2. Only call apply_discount AFTER you have received "  # Rule 2
                "a price from get_product_price. Pass the exact price "  # Continuation
                "returned by get_product_price - do NOT pass a made-up number.\n"  # Continuation
                "3. NEVER calculate discounts yourself using math. "  # Rule 3
                "Always use the apply_discount tool.\n"  # Continuation
                "4. If the user does not specify a discount tier, "  # Rule 4
                "ask them which tier to use - do NOT assume one."  # Continuation
            )
        ),
        HumanMessage(content=question)  # Add the human's question as a message
    ]
    for iteration in range(1, MAX_ITERATIONS+1):  # Loop for iterations
        print(f"Iteration {iteration} - LLM generating response...")  # Print iteration info
        ai_message = llm_with_tools.invoke(messages)  # Invoke the LLM with tools
        tool_calls = ai_message.tool_calls  # Get tool calls from the message
        if not tool_calls:  # If no tool calls
            print(f"\n Final Answer: {ai_message.content}")  # Print final answer
            return ai_message.content  # Return the answer
#        Process only the FIRST tool call -- force one tool per iteration  # Comment for processing tool calls
        tool_call = tool_calls[0]  # Get the first tool call
        tool_name = tool_call.get("name")  # Get the tool name
        tool_args = tool_call.get("args")  # Get the tool arguments
        tool_call_id = tool_call.get("id")  # Get the tool call ID

        print(f" [Tool Selected] {tool_name} with args: ({tool_args})")  # Print selected tool

        tool_to_use = tools_dict.get(tool_name)  # Get the tool function
        if tool_to_use is None:  # If tool not found
            raise ValueError(f"Tool {tool_name} not found in tools list")  # Raise error
        observation = tool_to_use.invoke(tool_args)  # Invoke the tool

        print(f" [Tool Result] {observation}")  # Print tool result

        messages.append(ai_message)  # Append AI message to messages
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))  # Append tool message


if __name__ == "__main__":  # If this script is run directly
    print("Hello Langchain Agent (.bind_tools)")  # Print greeting
    result = run_agent("What is the price of a laptop applying a gold discount?")  # Run the agent
    print(result)  # Print the result
