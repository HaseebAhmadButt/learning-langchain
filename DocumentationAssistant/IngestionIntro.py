# Optional Tutorial for TavilyMap and TavilyExtract
# https://nisum.udemy.com/course/langchain/learn/lecture/51313845/?udfrontends=true&cteMode=standalone

# Optional for TavilyMap to extract and use Batch processing for scraping the webpages.
# https://nisum.udemy.com/course/langchain/learn/lecture/51359991?udFrontends=false#content
import asyncio
import os, ssl
from typing import Any, Dict, List
import certifi
from dotenv import load_dotenv
from langchain_classic import text_splitter
from langchain_classic.agents.agent_toolkits import vectorstore

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

# Load environment variables from .env file
load_dotenv()

# Load SSL certificate for HTTPS requests
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUEST_CA_BUNDLE"] = certifi.where()

# Define Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=True,
    # Setting chunk_size=50 means LangChain will send exactly 50 texts to the
    # OpenAI API in a single HTTP request, wait for the embeddings, and then
    # send the next 50.
    chunk_size=50,
    # After every failure, minimum wait for the next request.
    retry_min_seconds=10
)

# chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(index_name="langchain-doc-index", embedding=embeddings)
tavilY_extract = TavilyExtract()
tavilY_map = TavilyMap(max_depth=5,
                       max_breadth=20,
                       max_pages=1000)
tavily_crawl = TavilyCrawl()



async def main():
    """Main async function to orchestrate the entire process."""
    # Invoke triggers the Tavily crawler tool, passing in a dictionary of configuration settings.
    # The result of this crawl (the extracted text, metadata, and URLs) will be stored in 'response'.
    response = tavily_crawl.invoke({

        # url: The starting point or "root" destination for the crawler.
        # The crawler will begin by visiting the LangChain Python documentation homepage.
        "url": "https://python.langchain.com",

        # max_depth: Controls how many layers of internal links the crawler is allowed to follow.
        # A depth of 5 means it will scrape the root URL (Level 0), find links on that page to scrape (Level 1),
        # find links on those pages (Level 2), and so on, up to 5 clicks away from the root.
        "max_depth": 1,
        # extract_depth: Dictates the method and thoroughness of the data extraction on each page.
        # Setting this to "advanced" tells Tavily to perform a high-quality scrape, which typically
        # includes rendering dynamic JavaScript, bypassing basic anti-bot walls, and returning cleaner,
        # more structured markdown or text compared to a basic HTML pull.
        "extract_depth": "advanced",
        # This will tell crawler to select which page and to ignore which page.
        "instructions": "content on ai agents"
    })
    all_docs = response["results"]
    print(f"Total documents extracted: {len(all_docs)}")
    # Access the content from the dictionary (Tavily returns dicts, not Document objects)
    langChainDocuments = []
    for doc in all_docs:
        langChainDocuments.append(Document(page_content=doc["raw_content"], metadata={"source": doc["url"]}))
    # Now you have a list of Document objects that you can use for embedding and vector storage.
    print(f"First document content preview: {langChainDocuments[0].page_content[:500]}")  # Print the first 500 characters of the first document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(langChainDocuments)


if __name__ == "__main__":
    asyncio.run(main())
