# ChatPromptTemplate is going to hold our content that we either sent to the
# LLM as humans, or that we receive back from the LLM as an answer
# that is tagged as an AI.

# MessagesPlaceholder is going to give us flexibility to put here a placeholder for future messages
# that we are going to get.

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generating critique and recommendation for the user's tweet"
            "Always provide detailed recommendations, including requests for length, virality, style, etc."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            "Generate the best twitter post possible for the user's request."
            "If the user provides critique, response with a revised version of your previous attempts"
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)


llm = ChatOpenAI(temperature=0)
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
