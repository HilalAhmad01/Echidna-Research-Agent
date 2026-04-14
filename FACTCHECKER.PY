from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import sys
import os



load_dotenv()

class FactCheckResult(BaseModel):
    fact_checked: str = Field(description="The fact-checked version of the text")
    accuracy: str = Field(description="The accuracy level (e.g., accurate, partially accurate, inaccurate)")

fact_check_parser = PydanticOutputParser(pydantic_object=FactCheckResult)

prompt = ChatPromptTemplate.from_template(
    "You are a professional fact-checker. Check the accuracy of the text provided below.\n"
    "{format_instructions}\n"
    "Text to fact-check: {input_text}"
).partial(format_instructions=fact_check_parser.get_format_instructions())

def main():
    llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.2,)
    

    # Create the Chain using LCEL (LangChain Expression Language)
    chain = prompt | llm | fact_check_parser

    print("--- Fact-Check Tool (Powered by Google Gemini)---")
    
    try:
        # Single Input
        user_input = input("Paste your paragraph and press Enter to fact-check:\n> ")

        if user_input.strip():
            print("\nFact-checking your text with Google Gemini... Please wait.\n")
            
            # Execute the Chain
            response = chain.invoke({"input_text": user_input})
            print("Fact-Checked Result:")
            print(f"Fact-Checked Text: {response.fact_checked}")
            print(f"Accuracy: {response.accuracy}")
        else:
            print("No input provided. Please paste a paragraph to fact-check.")
    except Exception as e:
        print(f"An error occurred: {e}")
if __name__ == "__main__":
    main()

