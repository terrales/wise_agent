import json
import numpy as np
import nltk
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.tools import tool

nltk.download('punkt')


class Memory:
    def __init__(self):
        self.memory_store = {}  # Memory dictionary to store key-value pairs
        self.key_embeddings = None  # NumPy array to store key embeddings
        self.keys = []  # List to keep track of the keys (index aligned with key_embeddings)
        
        # Embedding model initialization
        self.model_name = "BAAI/bge-small-en"
        self.model_kwargs = {"device": "cpu"}
        self.encode_kwargs = {"normalize_embeddings": True}
        self.hf = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )

def exact_match_query(memory: Memory, key):
    """
    Retrieve the value for an exact key match.
    """
    return memory.memory_store.get(key, None)


def query_similar_keys(memory: Memory, query, k=2):
    """
    Retrieve the top-k most similar keys based on the query embedding.
    """
    if not memory.memory_store:
        return []

    # Embed the query
    query_embed = memory.hf.embed_query(query)
    
    # Calculate similarity scores
    scores = np.dot(memory.key_embeddings, query_embed)
    top_k_idx = np.argpartition(scores, -k)[-k:]
    top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]

    # Return the top K similar keys and their values
    return [
        {"key": memory.keys[idx], "value": memory.memory_store[memory.keys[idx]], "similarity": scores[idx]}
        for idx in top_k_idx_sorted
    ]

def retrieve_from_memory(key: str, memory: Memory):
    """
    Given a query, retrieve the answers to similar queries from memory.

    Args: 
    key: the original user query and if it was ambuiguous or unambiguous. 
            use this format: f"{user query} - {ambiguity type}".
    memory: Instance of the Memory class.
    """
    if exact_match_query(memory, key) is not None:
        return exact_match_query(memory, key)
    else:
        return query_similar_keys(memory, key)


def update_memory(query: str, steps_to_answer: dict, memory: Memory):
    """
    Update or insert a key-value pair in memory and update key embeddings.

    Args: 
    query: the string containing the original user query, and if it was ambuiguous or unambiguous. 
            use this format: f"{user query} - {ambiguity type}"
    steps_to_answer (Dict{steps: list, answer: str}): the dictionary containing the list of tools 
                                    called to answer the query and the string containing the answer.
    memory: Instance of the Memory class.
    """
    if query in memory.memory_store:
        # Update value, embedding remains the same
        memory.memory_store[query] = steps_to_answer
    else:
        # Add new key-value pair
        memory.memory_store[query] = steps_to_answer
        memory.keys.append(query)
        
        # Generate and append the key embedding
        key_embedding = memory.hf.embed_query(query)
        if memory.key_embeddings is None:
            memory.key_embeddings = np.array([key_embedding])
        else:
            memory.key_embeddings = np.vstack([memory.key_embeddings, key_embedding])


def save_to_file(memory: Memory, file_path):
    """
    Save memory to a file.
    """
    with open(file_path, 'w') as f:
        json.dump(memory.memory_store, f, indent=4)


def load_from_file(memory: Memory, file_path):
    """
    Load memory from a file and regenerate key embeddings.
    """
    with open(file_path, 'r') as f:
        memory.memory_store = json.load(f)

    # Reinitialize keys and embeddings
    memory.keys = list(memory.memory_store.keys())
    memory.key_embeddings = np.array([memory.hf.embed_query(key) for key in memory.keys])