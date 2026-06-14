import datetime

from dotenv import load_dotenv
load_dotenv()

from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser, PydanticToolsParser
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from schemas import AnswerQuestion, ReviseAnswer

llm = ChatOpenAI(model="gpt-4o",temperature=0)
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an expert researcher.
            Count time: {time}
            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvements.
            3. Recommended search queries to research information and improve your answer.
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Answer the user's question above using the required format."
        )
    ]
).partial(
    time=lambda: datetime.datetime.now().strftime("%H:%M:%S")
)

first_responder_prompt = actor_prompt_template.partial(first_instruction="""
Provide a detailed ~250 word answer
""")

first_responder = (first_responder_prompt
                   | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion" # Force the LLM to use the AnswerQuestion tool
))

revise_instruction = """Revise your previous answer using the new information.
- You should use the previous critique to add important information to your answer.
    - You MUST include numerical citations in you revised answer to ensure it can be verified.
    - Add a 'References' section to the bottom of your answer (which does not count towards the word limit the form of):
        - [1] https://example.com
        - [2] https://example.com
    - You should use the previous critique to remove the superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor = actor_prompt_template.partial(
    first_instruction=revise_instruction
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="auto")

if __name__ == "__main__":
    human_message = HumanMessage(content="""
    Write about AI-Powered SOC / autonomous soc problem domain,
    list starups that do that and raised capital
    """)

    chain = (
        first_responder_prompt
        | llm.bind_tools(tools=[AnswerQuestion], tools_choice="auto")
        | parser.pydantic_parse
    )

    resp = chain.invoke(input=human_message)
    print(resp)