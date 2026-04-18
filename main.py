# This project is developed using "pip3" package manager.

# Import necessary modules
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
# Import modules: dotenv is imported to load environment 
# variables from a .env file, and os is imported to access 
# those variables.

# Load environment variables: load_dotenv() reads the .env 
# file (if present) and sets the variables in the system's environment.
load_dotenv()

# # Main function
# def main():
#     # Print a greeting message
#     print("Hello, World!")
#     # Print the OpenAI API key from environment variables
#     print(os.environ.get("OPEN_API_KEY"))


def main():
    information = """Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, X, and xAI. Musk has been the wealthiest person in the world since 2025; as of February 2026, Forbes estimates his net worth to be around US$852 billion.
       Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.
      In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research, but later left; growing discontent with the organization's direction and leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired the social network Twitter, implementing significant changes, and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017. In November 2025, a Tesla pay package worth $1 trillion for Musk was approved, which he is to receive over 10 years if he meets specific goals.
      Musk was the largest donor in the 2024 U.S. presidential election, where he supported Donald Trump. After Trump was inaugurated as president in early 2025, Musk served as Senior Advisor to the President and as the de facto head of the Department of Government Efficiency (DOGE). After a public feud with Trump, Musk left the Trump administration and returned to managing his companies. Musk is a supporter of global far-right figures, causes, and political parties. His political activities, views, and statements have made him a polarizing figure. Musk has been criticized for COVID-19 misinformation, promoting conspiracy theories, and affirming antisemitic, racist, and transphobic comments. His acquisition of Twitter was controversial due to a subsequent increase in hate speech and the spread of misinformation on the service, following his pledge to decrease censorship. His role in the second Trump administration attracted public backlash, particularly in response to DOGE. The emails he sent to Jeffrey Epstein are included in the Epstein files, which were published between 2025–26 and became a topic of worldwide debate."""
    
    # This is the template for the prompt that will be used to generate a summary and interesting facts about a person based on the provided information. The template includes placeholders for the input variable "information" which will be filled with the actual information about the person when the prompt is used.
    summary_template = """
    given the information {information}, about a person I want you to create:
    1. A short summary.
    2. two interesting facts about them.
    """

    # Using the PromptTemplate class to create a prompt template
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    # Create an instance of the ChatOpenAI class to interact
    # temperature will control:
    # How random or creative versus strict and deterministic the model's responses will be. A higher temperature (e.g., 0.8) will result in more creative and varied responses, while a lower temperature (e.g., 0.2) will produce more focused and consistent answers.
    llm = ChatOpenAI(model="gpt-5", temperature=0)

    # This is an example LangChain Expression Language (LCEL) expression that creates a chain of operations. The summary_prompt_template is combined with the llm (language model) to create a chain that will generate a response based on the provided information. The chain is then invoked with the input variable "information" to produce the desired output.
    # We have composed two components a prompt template and a LLM to create a chain that will generate a response based on the provided information.
    # Output of the left component (summary_prompt_template) will be passed as input to the right component (llm).
    # The resulting chain is a Runnable Object, of Runnable Interface. This object can be invoked with the input variable "information" to produce the desired output, which will be a short summary and two interesting facts about the person based on the provided information.
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})

    # Datatype of response is "AIMessage". To get the content of the response, we can access the "content" attribute of the response object.
    # The chat id can be accessed using the "id" attribute of the response object.
    # To get the usage of the response, we can access the "usage_metadata" attribute of the response object, which contains information about the number of tokens used in the input and output.
    print(response.content)


# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()
