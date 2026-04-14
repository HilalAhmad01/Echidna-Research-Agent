import gradio as gr
import os
from dotenv import load_dotenv

# --- Local Ollama & LangChain Imports ---
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from tavily import TavilyClient
from pydantic import BaseModel, Field

# Still imported in case you want to use it, but logic below uses Ollama
from langchain_google_genai import ChatGoogleGenerativeAI 

load_dotenv()

# --- Structured Data Models ---
class FactCheckResult(BaseModel):
    fact_checked: str = Field(description="The fact-checked version of the text")
    accuracy: str = Field(description="The accuracy level (e.g., accurate, partially accurate, inaccurate)")

class ParaphraseResult(BaseModel):
    paraphrased: str = Field(description="The newly paraphrased version of the text")
    tone: str = Field(description="The tone detected or used (e.g., formal, casual)")

# --- Logic Functions ---

def build_research_chain(user_query, search_type="advanced", max_results=5):
    search_type_map = {"basic": "basic", "advanced": "advanced"}
    client = TavilyClient(os.getenv("TAVILY_API_KEY"))
    response = client.search(
        query=user_query,
        search_depth=search_type_map.get(search_type, "advanced"),
        max_results=max_results,
        include_raw_content="text",
    )
    
    articles = response.get("results", [])
    final_sources = [(a.get("title"), a.get("url")) for a in articles]
    content_list = [a.get("raw_content") or a.get("content", "") for a in articles]
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents(content_list)
    
    # Local Embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Local Chat Model
    chat_model = ChatOllama(model="gemma2:2b", temperature=0.3)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer ONLY from the provided context. If the context is insufficient, say you don't know."),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])
    
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        RunnableParallel({"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()})
        | prompt | chat_model | StrOutputParser()
    )
    
    return chain, final_sources

def fact_check_action(text):
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.2)
    parser = PydanticOutputParser(pydantic_object=FactCheckResult)
    prompt = ChatPromptTemplate.from_template(
        "Fact-check this: {input_text}\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())
    
    chain = prompt | llm | parser
    res = chain.invoke({"input_text": text})
    return res.fact_checked, res.accuracy

def paraphrase_action(text):
    llm = ChatOllama(model="gemma2:2b", temperature=0.7)
    parser = PydanticOutputParser(pydantic_object=ParaphraseResult)
    prompt = ChatPromptTemplate.from_template(
        "You are a professional writing assistant. Paraphrase this text and detect the tone.\n"
        "{format_instructions}\n"
        "Text: {input_text}"
    ).partial(format_instructions=parser.get_format_instructions())
    
    chain = prompt | llm | parser
    res = chain.invoke({"input_text": text})
    return res.paraphrased, res.tone

# --- Gradio UI ---

custom_css = """
body { background-color: #0f172a; color: #f8fafc; }
.gradio-container { border-radius: 20px; background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); }
button.primary { background: linear-gradient(90deg, #6366f1, #a855f7) !important; border: none !important; }
"""

with gr.Blocks() as demo:
    gr.Markdown("# 🌌 Echidna AI Research Station")
    current_chain = gr.State()
    
    with gr.Tabs():
        with gr.Tab("🔍 Deep Research"):
            with gr.Row():
                topic_input = gr.Textbox(label="Research Topic", placeholder="e.g. History of Linux Mint")
                search_depth = gr.Radio(["basic", "advanced"], label="Search Depth", value="advanced")
                num_res = gr.Slider(1, 10, value=5, step=1, label="Sources")
            
            research_btn = gr.Button("Initialize Research Pipeline", variant="primary")
            
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Chat with Research Context", height=400)
                    chat_input = gr.Textbox(label="Ask a question", placeholder="Type here...")
                    chat_btn = gr.Button("Ask")
                with gr.Column(scale=1):
                    sources_display = gr.HTML(label="Sources Found")

            def start_research(topic, depth, num):
                chain, sources = build_research_chain(topic, depth, num)
                source_html = "<ul>" + "".join([f"<li><a href='{s[1]}' target='_blank'>{s[0]}</a></li>" for s in sources]) + "</ul>"
                return chain, source_html, []

            def chat_fn(question, history, chain):
                if chain is None:
                    history.append({"role": "user", "content": question})
                    history.append({"role": "assistant", "content": "Please initialize research first!"})
                    return "", history
                
                history.append({"role": "user", "content": question})
                try:
                    ans = chain.invoke(question)
                except Exception as e:
                    ans = f"Error: {str(e)}"
                
                history.append({"role": "assistant", "content": ans})
                return "", history

            research_btn.click(start_research, [topic_input, search_depth, num_res], [current_chain, sources_display, chatbot])
            chat_input.submit(chat_fn, [chat_input, chatbot, current_chain], [chat_input, chatbot])
            chat_btn.click(chat_fn, [chat_input, chatbot, current_chain], [chat_input, chatbot])

        with gr.Tab("⚖️ Fact Checker"):
            fc_input = gr.Textbox(label="Text to Verify", lines=5)
            fc_btn = gr.Button("Verify with Gemini3", variant="primary")
            with gr.Row():
                fc_output = gr.Textbox(label="Verified Text")
                fc_accuracy = gr.Label(label="Accuracy Confidence")
            fc_btn.click(fact_check_action, fc_input, [fc_output, fc_accuracy])

        with gr.Tab("✍️ Paraphraser"):
            p_input = gr.Textbox(label="Original Text", lines=5)
            p_btn = gr.Button("Rewrite with Gemma 2", variant="primary")
            with gr.Row():
                p_output = gr.Textbox(label="Paraphrased Version")
                p_tone = gr.Textbox(label="Detected Tone")
            p_btn.click(paraphrase_action, p_input, [p_output, p_tone])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=custom_css)