# In this file we are going to the Model Integration Clients. This file contains the code for:

# ===================================== Raw Function Calling =========================================

# Raw Function Calling means that we pass the complete function details to the LLM, through provided schema of the Model.
# Ollama accepts the simple Python function and can use these as tools, but if we want to use other LLMs, we will have to provide the function schema in the required format.

from dotenv import load_dotenv  # Import the load_dotenv function to load environment variables from a .env file
load_dotenv()  # Execute load_dotenv to load the environment variables

import ollama
from langsmith import traceable  # Import traceable for tracing

# Without @tool, we must manually define the JSON Schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.
# For the Ollama the calling schema can be found here: https://docs.ollama.com/capabilities/tool-calling#tool-calling
# If we shift to Anthropic, we will have to use its schema.

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g, 'laptop', 'smartphone', 'headphones'"
                    }
                },
                "required": ["product"]
            }
        }
    },
{
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The original price"
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier, e.g, 'bronze', 'silver', 'gold'"
                    }
                },
                "required": ["price", "discount_tier"]
            }
        }
    }
]

# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
# tools_for_llm = [get_product_price, apply_discount]
# However, this required the functions docstring to follow Googles docstring format.
# So, Ollama can parse the parameter descriptions from the Args section.



MAX_ITERATIONS = 10  # Define the maximum number of iterations for the agent loop
MODEL_NAME = "qwen3-coder:30b"  # Define the model name for the LLM


# While using the LangChain, it automatically proved the traceability of each of the tools, but when we move to Raw Function Calling
# We have to manually trace each of the functions.
@traceable(run_type="tool")
def get_product_price(product: str) -> str:  # Define a tool function to get product price
    """Look up the price of a product in the catalog"""  # Docstring for the function
    print(f"  >> Executing get_product_price(product='{product}')")  # Print execution message
    prices: dict[str, int] = {  # Define a dictionary of product prices
        "laptop": 999,  # Price for laptop
        "smartphone": 499,  # Price for smartphone
        "headphones": 199  # Price for headphones
    }
    return prices.get(product.lower(), 0)  # Return the price or 0 if not found

@traceable(run_type="tool")
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


# When we use LangChain it directly traces this under the hood, but when we use Raw Function Calling, we have to manually trace it.
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):  # Define the main agent function
    return ollama.chat(model=MODEL_NAME, messages=messages, tools=tools_for_llm)

#   ------- Agent Loop -------

@traceable(name="Langchain Agent Loop")  # Decorator for tracing the function
def run_agent(question: str):  # Define the main agent function
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }

    print(f"Question: '{question}')")  # Print the user's question
    print("=" * 60)  # Print a separator line

    messages = [  # Instead of using specified classes for messages, we will use a simple list of dicts with "role" and "content". This is the structure that Ollama supports.
        # For other models we will have to use their provided messages structure or classes.
        {
            "role": "system",
            "content": (  # Content of the system message
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
        },
        {"role": "user", "content": question}
    ]
    for iteration in range(1, MAX_ITERATIONS+1):  # Loop for iterations
        print(f"Iteration {iteration} - LLM generating response...")  # Print iteration info

        response = ollama_chat_traced(messages=messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        if not tool_calls:  # If no tool calls
            print(f"\n Final Answer: {ai_message.content}")  # Print final answer
            return ai_message.content  # Return the answer
#        Process only the FIRST tool call -- force one tool per iteration  # Comment for processing tool calls
        tool_call = tool_calls[0]  # Get the first tool call, this gives the tool to be called
        tool_name = tool_call.function.name # Get Tool Name selected by the LLM
        tool_args = tool_call.function.arguments # Get list of arguments and their values for the selected tool


        print(f" [Tool Selected] {tool_name} with args: ({tool_args})")  # Print selected tool

        tool_to_use = tools_dict.get(tool_name)  # Extract its implementation from the tools_dict

        if tool_to_use is None:  # If tool not found
            raise ValueError(f"Tool {tool_name} not found in tools list")  # Raise error

        observation = tool_to_use(**tool_args) # Invoke the tool

        print(f" [Tool Result] {observation}")  # Print tool result

        messages.append(ai_message)  # Append the previous AI response to the message array to maintain context
        # Add a new context message for the tool call, and its result
        messages.append(
            {
                "role": "tool",
                "content": str(observation)
            }
        )  # Append tool message
    return None


if __name__ == "__main__":  # If this script is run directly
    print("Hello Langchain Agent (.bind_tools)")  # Print greeting
    result = run_agent("What is the price of a laptop applying a gold discount?")  # Run the agent
    print(result)  # Print the result
