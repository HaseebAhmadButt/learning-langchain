# Retrieving the matching document from the database
# This link from LangChain represents the combination of these two documents for building the semantic systems:
# https://docs.langchain.com/oss/python/langchain/knowledge-base

import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

from langchain_classic.chains.summarize.map_reduce_prompt import prompt_template
from langchain_classic.schema import retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv() # Load environment variables from .env file

embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
llm = ChatOpenAI(model="gpt-5.2",openai_api_key=os.environ["OPENAI_API_KEY"])

vectorStore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
retriever = vectorStore.as_retriever(search_kwargs={"k": 3}) # {"k": 3} ===> Represents the top 3 documents

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    
    Provide a detailed answer: 
    """
)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join([doc.page_content for doc in docs])


# ==========================================================
# IMPLEMENTATION 1: Without LCEL (Simple Functional-Based Approach)
# ==========================================================
def retrieval_chain_without_lcel(query: str): # lcel: LangChain Expression Language
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    -- Manually step-by-step execution.
    -- No built-in streaming support.
    -- No async support without additional code.
    -- Harder to compose with other chains.
    -- More verbose and error-prone.
    """
    # Step 1: Retrieve relevant documents based on the query
    doc = retriever.invoke(query)

    # Step 2: Format the retrieved documents into a single string to be used as context for the LLM
    context = format_docs(doc)
    print("\n" + "=" * 70)
    print("Document Returned from Pinecone: ")
    print(context)
    print("\n" + "=" * 70)

    # Step 3: Create the prompt by filling in the context and question into the prompt template
    prompt = prompt_template.format_messages(context=context, question=query)

    # Step 4: Generate the response using the LLM
    response = llm.invoke(prompt)

    # Step 5: Return the response content
    return response.content


def create_retrieval_chain():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    -- Declarative and composable: Easy to chain operations with pipe operator (|)
    -- Built-in streaming: chain.stream() works out of the box
    -- Built-in async: chain.invoke() and chain.stream() available
    -- Batch Processing: chain.batch() for multiple inputs
    -- Type safety: Better integration with LangChain's type system
    -- Less code: More concise and readable
    -- Reusable: Chain can be saved, shared, and composed with other chains.
    -- Better debugging: LangChain provides better observability tools.
    """
    retrieval_chain = (
       # This represents the five steps of the retrieval chain, as mentioned in the above method.
       # format_docs is a simple function and is not a Runnable.
       # Other parameters are Runnables, which are functions that return Runnable.
       # For these kinds of functions, LangChain automatically wraps them in RunnableLambda; for this case it is RunnableLambda(format_docs).
       # This represents the Runnable Interface in LangChain.
       # Now, the problem is that prompt_template needs two parameters: context and question.
       # This implementation will not work with that. So, we are going to use new approach.
       #  =================================================================
       # retriever | format_docs| prompt_template | llm | StrOutputParser()
       # ==================================================================

    #     New Approach is:
        RunnablePassthrough.assign(
            # Get the question from input, retrieve documents, and format them as context
            # This will create a dictionary with the question and the formatted documents.
            context=itemgetter("question") | retriever | format_docs,
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain



if __name__ == "__main__":
    print("Retrieving......")

    # Query
    query = "What is Pinecone in Machine Learning?"

    # ===================================================
    # Option 0: Raw invocation without 3_RAG
    # ===================================================

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation without 3_RAG")
    print("="*70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer: ")
    print(result_raw.content)

    # ===================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ===================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer: ")
    print(result_without_lcel)

    # ===================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ===================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better than the previous approach:")
    print("-- More concise and declarative.")
    print("-- Built-in streaming: chain.stream()")
    print("-- Built-in async: chain.ainvoke")
    print("-- Easy to compose with other chains.")
    print("Better for production use.")
    print("=" * 70)
    chain_with_lcel = create_retrieval_chain()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer: ")
    print(result_with_lcel)
