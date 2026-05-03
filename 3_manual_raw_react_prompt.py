# Raw Function Calling through ReAct Prompt.
import re  # Regular Expressions library provided by Python
import inspect  # Inspect live objects --> Python library

from dotenv import load_dotenv  # Import the load_dotenv function to load environment variables from a .env file

load_dotenv()  # Execute load_dotenv to load the environment variables

import ollama
from langsmith import traceable  # Import traceable for tracing

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
    price = float(price)  # Ensure price is a float, because by default LLM output in text and this needs to be cast to float
    discount_percentage: dict[str, float] = {  # Define a dictionary of discount percentages
        "bronze": 5,  # Bronze discount percentage
        "silver": 12,  # Silver discount percentage
        "gold": 23  # Gold discount percentage
    }
    discount = discount_percentage.get(discount_tier, 0)  # Get the discount percentage or 0
    return round(price * (1 - discount / 100), 2)  # Calculate and return the discounted price rounded to 2 decimals




def get_tool_description(tool_dict):
    descriptions = []
    for tool_name, tool_function in tool_dict.items():
        # __wrapped__ bypass the decorator wrappers (e.g, @traceable) and gets the original function to extract its signature and docstring for the description.
       original_function = getattr(tool_function, "__wrapped__", tool_function)
       signature = inspect.signature(original_function)  # Parameters this function accepts
       docstring = inspect.getdoc(original_function)     # Docstring for the function
       descriptions.append(f"{tool_name}({signature}): {docstring}")
    return "\n".join(descriptions)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}

tool_descriptions = get_tool_description(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f""""
STRICT RULES - you must follow these exactly:
"1. NEVER guess or assume any product price. " 
"You MUST call get_product_price first to get the real price.\n" 
"2. Only call apply_discount AFTER you have received " 
"a price from get_product_price. Pass the exact price " 
"returned by get_product_price - do NOT pass a made-up number.\n"
"3. NEVER calculate discounts yourself using math. "
"Always use the apply_discount tool.\n"
"4. If the user does not specify a discount tier, "
"ask them which tier to use - do NOT assume one."
        
Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:

"""""

# When we use LangChain it directly traces this under the hood, but when we use Raw Function Calling, we have to manually trace it.
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):  # Define the main agent function
    return ollama.chat(model=model, messages=messages, options=options)




#   ------- Agent Loop -------

@traceable(name="Langchain Agent Loop")  # Decorator for tracing the function
def run_agent(question: str):  # Define the main agent function
    print(f"Question: '{question}')")  # Print the user's question
    print("=" * 60)  # Print a separator line

    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS + 1):  # Loop for iterations
        print(f"Iteration {iteration} - LLM generating response...")  # Print iteration info

        full_prompt = prompt + scratchpad

        response = ollama_chat_traced(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0}
        )

        output = response.message.content
        print(f"LLM Output: \n{output}")

        print(f" [Parsing] Looking for Final Answer in LLM Output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f" [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        print(f" [Parsing] Looking for Action and Action Input in LLM Output...")
        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match : # If either Action or Action Input is not found, raise an error
            print(
                " [Parsing] Action or Action Input not found in LLM Output. Exiting the loop.[]"
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f" [Tool Selected] {tool_name} with args: ({tool_input_raw})")  # Print selected tool
        # Split comma-separated args; strip key= prefix if LLM outputs key-value format
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split(":", 1)[-1].strip("}").strip().strip("'\"") for x in raw_args]

        print(f" [Tool Executing] {tool_name}({args})...")

        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list[str](tools.keys())}"

        else:
            observation = str(tools[tool_name](*args))

        print(f" [Tool Result] {observation}")  # Print tool result
        scratchpad += f"{output}\nObservation: {observation}\nThought: "  # Append the new output and observation to the scratchpad for the next iteration
    return None


if __name__ == "__main__":  # If this script is run directly
    print("Hello Langchain Agent (.bind_tools)")  # Print greeting
    result = run_agent("What is the price of a laptop applying a gold discount?")  # Run the agent
    print(result)  # Print the result