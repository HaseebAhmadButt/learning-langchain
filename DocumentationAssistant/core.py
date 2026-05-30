import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
    show_progress_bar=True,
    # Setting chunk_size=50 means LangChain will send exactly 50 texts to the
    # OpenAI API in a single HTTP request, wait for the embeddings, and then
    # send the next 50.
    chunk_size=50,
    # After every failure, minimum wait for the next request.
    retry_min_seconds=10,
    dimensions=512,
    openai_api_key=os.environ["OPENAI_API_KEY"])
vectorstore = PineconeVectorStore(index_name="langchain-doc-index", embedding=embeddings)
model = init_chat_model("gpt-5.2", model_provider="openai", openai_api_key=os.environ["OPENAI_API_KEY"])


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    retrieve_docs = vectorstore.as_retriever().invoke(query, k=4)
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get("source", "")}\n\nContent: {doc.page_content}"
                             for doc in retrieve_docs)

    return (
            serialized, # Passed down to LLM as context
            retrieve_docs # Used by application
            )

# This will run the RAG pipeline to answer a query using retrieved documentation.
# The agent automatically executes the steps we implemented in previous RAG files to implement the RAG pipeline when we call agent.invoke(query).
# It automatically populates the system message and calls the retrieve_context tool to retrieve relevant documentation.
# The retrieved documentation is then used as context to generate a response to the user's query.
# This is like a short and automated version of the RAG pipeline.
def run_llm(query: str):
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    Args:
        query:

    Returns:
        Dictionary containing:
         - answer: The generated answer.
         - context: List of retrieved documents.
    """

    system_prompt = (
        "You are a helpful assistant that answers questions about the LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation."
        "Use the tool to find relevant information before answering questions."
        "Always cite the sources you use in your answers."
        "If you cannot find the answer in the retrieved documentation, say so."
    )
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
    messages = [{
        "role": "user",
        "content": query
    }]
    response = agent.invoke({
        "messages": messages
    })
    answer = response["messages"][-1].content

    context_docs = []
    for message in response["messages"]:
        if isinstance(messages, ToolMessage) and hasattr(message, "artifact"):
            context_docs.append(message.artifact)


    return {
        "answer": answer,
        "context": context_docs
    }


if __name__ == "__main__":
    result = run_llm("What is LangChain Agent Architecture?")
    print(result)
