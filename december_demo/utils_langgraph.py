import getpass
import os
import re
import numpy as np
from langchain_core.tools import tool
import pandas as pd
import numpy as np
import nltk 
#nltk.download('punkt_tab')
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from nltk.tokenize import sent_tokenize
from tavily import TavilyClient
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.runnables import Runnable, RunnableConfig
from datetime import date, datetime
from langchain_mistralai import ChatMistralAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "test_langgraph"
os.environ["MISTRAL_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = ""
os.environ["LANGSMITH_API_KEY"] = ""

df = pd.read_json("hf://datasets/yixuantt/MultiHopRAG/MultiHopRAG.json")
corpus = pd.read_json("hf://datasets/yixuantt/MultiHopRAG/corpus.json")

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

loader = DataFrameLoader(corpus, page_content_column="body")
documents = loader.load()

# Helper function to split document body into sentences
def split_into_sentences(doc):
    return sent_tokenize(doc.page_content)

# VectorStoreRetriever class to store and query sentences
class VectorStoreRetriever:
    def __init__(self, docs: list, vectors: list, sentences: list):
        self._arr = np.array(vectors)  # Embedding vectors for sentences
        self._docs = docs  # Original documents
        self._sentences = sentences  # List of sentence-level content

    @classmethod
    def from_docs(cls, docs):
        all_sentences = []
        
        # Process each document: split it into sentences, generate embeddings
        for doc in docs:
            sentences = split_into_sentences(doc)  # Split into sentences
            all_sentences.extend(sentences)
        
        # Generate embeddings for the sentences
        sentence_embeddings = hf.embed_documents(all_sentences)
        return cls(docs, sentence_embeddings, all_sentences)

    def query(self, query: str, k: int = 5) -> list[dict]:
        # Embed the query
        embed = hf.embed_query(query)
        
        # Calculate similarity scores
        scores = np.array(embed) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        
        # Return the top K similar sentences
        return [
            {"sentence": self._sentences[idx], "similarity": scores[idx]}
            for idx in top_k_idx_sorted
        ]

# Initialize the retriever with sentence-level embeddings
retriever = VectorStoreRetriever.from_docs(documents)

@tool
def find_answer_in_docs(query: str) -> str:
    """do RAG"""
    # Perform query and get the most relevant sentences
    docs = retriever.query(query, k=4)
    
    # Return the retrieved sentences as a string
    retrieval_results = "\n\n".join([doc["sentence"] for doc in docs])
    return retrieval_results

@tool 
def search_web(query: str) -> str:
    """search the web"""
    tavily_client = TavilyClient(api_key="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")
    response = tavily_client.search(query)
    return response

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
         #   print("TYPE: ", message.type) #human, ai, tool
         #   print("CONTENT: ", message.content)
            msg_repr = message.pretty_repr(html=True)

            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)
            

def add_to_results(event: dict, results: dict):
    message = event.get("messages")
    if message:
         if isinstance(message, list):
            message = message[-1]
         if message.content != "":
            results[message.type] = message.content

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
        #    passenger_id = configuration.get("passenger_id", None)
            state = {**state}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}
    
def build_part_1_assistant_runnable():

    part_1_tools = [find_answer_in_docs, search_web]

    my_rate_limiter = InMemoryRateLimiter(
        requests_per_second = .08,
        check_every_n_seconds = 0.8,
        max_bucket_size = 5
    )
    
    llm2 = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0,
    max_retries=3,
    rate_limiter=my_rate_limiter,
    timeout = 240
    # other params...
    ).bind_tools(part_1_tools)

    primary_assistant_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a question answering assistant. "
                "Invoke the tools given to you when needed to respond to user queries."
                "Use the documents given to you for any questions on English magazines such as 'The Age', 'Fortune', 'The Verge' and 'TechCrunch'"
                " Provide a really short answer."
                " When searching, be persistent. Expand your query bounds if the first search returns no results. "
                " If a search comes up empty, expand your search before giving up."
                "All queries have an answer."
                "\nCurrent time: {time}.",
            ),
            ("placeholder", "{messages}"),
        ]
    ).partial(time=datetime.now)

    return primary_assistant_prompt | llm2

def initialize_part_1_graph():

    builder = StateGraph(State)
    part_1_tools =  [find_answer_in_docs, search_web]
    memory = MemorySaver()

    builder.add_node("assistant", Assistant(build_part_1_assistant_runnable()))
    builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    memory = MemorySaver()
    part_1_graph = builder.compile(checkpointer=memory)
    return part_1_graph



