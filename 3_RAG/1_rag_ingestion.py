# This document is about the Data Indexing where we store our
# data into the Vector space.

# It is like we are storing our documents in vector space that will
# help us to retrieve the relevant document.
from dotenv import load_dotenv
import os


load_dotenv()

# To understand this like Java or C++
# TextLoader is a class that loads text files, and document_loaders is package which contain other classes also.

from langchain_community.document_loaders import TextLoader # To load the text files
from langchain_text_splitters import RecursiveCharacterTextSplitter

# CharacterTextSplitter tries to split text using separators like:
#
# \n\n
# \n
# spaces
#
# If it cannot find a separator before reaching 1000 characters, it
# may produce a larger chunk.
# from langchain_text_splitters import CharacterTextSplitter # To split the text into chunks

from langchain_openai import OpenAIEmbeddings # OpenAI model to convert text into vector embeddings
from langchain_pinecone import PineconeVectorStore # To store the vector embeddings in Pinecone

if __name__ == "__main__":
    print("Hello World!")
    # Using Document Loader to load the text file, changing the loader, we can load different types of documents like PDF, HTML, Markdown, etc.
    loader = TextLoader(
        "/SampleDocument.txt",
                encoding="utf-8")

    documents = loader.load()

    print("Splitting....")
    # chunk_size=1000: Each chunk will be approximately 1000 characters long
    # chunk_overlap=0: There is no overlap between consecutive chunks -
    # they are split cleanly without repetition

    # Overlap is useful in 3_RAG (Retrieval-Augmented Generation) systems because:
    # Preserves Context: When you have chunk_overlap=200, for example, the last 200
    # characters of one chunk are repeated at the start of the next chunk. This preserves
    # important context that might be lost at chunk boundaries.

    # Better Retrieval: Important information that spans across chunk boundaries
    # won't get split awkwardly, improving search and retrieval accuracy.

    # CharacterTextSplitter tries to split text using separators like:
    #
    # \n\n
    # \n
    # spaces
    #
    # If it cannot find a separator before reaching 1000 characters, it
    # may produce a larger chunk.

    # text_splitter = CharacterTextSplitter(
    #     chunk_size=1000,
    #     chunk_overlap=0
    # )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    docs = text_splitter.split_documents(documents)
    print(f"created {len(docs)} chunks")

    # This step will produce the embeddings for the documents
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    print("ingesting....")
    PineconeVectorStore.from_documents(docs, embeddings, index_name=os.environ["INDEX_NAME"])

    print("Done!")