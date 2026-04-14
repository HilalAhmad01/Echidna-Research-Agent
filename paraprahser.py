from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import os

load_dotenv()

class ParaphraseResult(BaseModel):
    paraphrased: str = Field(description="The newly paraphrased version of the text")
    tone: str = Field(description="The tone detected or used (e.g., formal, casual)")

parser = PydanticOutputParser(pydantic_object=ParaphraseResult)

# Prompt matches the one in app.py for consistency
prompt = ChatPromptTemplate.from_template(
    "You are a professional writing assistant. Paraphrase the text provided below and detect the tone.\n"
    "{format_instructions}\n"
    "Text to paraphrase: {input_text}"
).partial(format_instructions=parser.get_format_instructions())

def main():
    # Initialize local Gemma 2
    chat_model = ChatOllama(model="gemma2:2b", temperature=0.7)
    chain = prompt | chat_model | parser

    print("\n" + "="*40)
    print("--- Local Gemma 2 Paraphrase Tool ---")
    print("="*40 + "\n")
    
    try:
        user_input = input("Paste your text and press Enter to paraphrase (or type 'exit' to quit):\n> ")

        if user_input.lower() == 'exit':
            return

        if user_input.strip():
            print("\nRefining your text locally via Ollama... Please wait.\n")
            
            response = chain.invoke({"input_text": user_input})
            
            print("="*40)
            print(f"PARAPHRASED VERSION:\n\n{response.paraphrased}")
            print(f"\nDETECTED TONE: {response.tone}")
            print("="*40)
        else:
            print("Error: You didn't enter any text.")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()