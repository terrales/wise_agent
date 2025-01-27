from langchain_core.tools import tool
import pandas as pd
from data_loader import data

class Evaluate:
    def __init__(self):
        pass

@tool
def simple_evaluate(answer: str):
    """
    this function evaluates if your answer is correct
    Args:
        answer: the answer to evaluate
    """
    return f"{answer} is the correct answer"


df = data

@tool
def validate_answer(sentence: str, continuation: str, agent_answer: str, clarification: str = ""):
    """
    Validates whether the agent's yes/no answer for the continuation is correct.

    Parameters:
        sentence (str): The input sentence.
        continuation (str): The continuation provided.
        agent_answer (str): your yes/no answer to whether the continuation is correct.
        clarification (str or None): An optional clarification.

    Returns:
        str: 'Correct' if the answer matches the dataset, 'Incorrect' otherwise.
    """
    # Normalize the user_answer to lowercase for consistency
    sentence = sentence.rstrip(".")
    clarification = clarification.rstrip(".")
    continuation = continuation.rstrip(".")
    agent_answer = agent_answer.rstrip(".")
    agent_answer = agent_answer.lower()

    # Validate input for user_answer
    if agent_answer not in ["yes", "no"]:
        return "Invalid answer. Please provide 'yes' or 'no'."

    # If clarification is provided, validate directly against the continuation
    if clarification is not "":
        sentence = clarification
    
    row = df[(df["control sentence 1"] == sentence) | (df["control sentence 2"] == sentence) | (df["underspecified sentence"] == sentence)] 

    if row.empty:
        print(sentence)
        print(clarification)
        return "Sentence or clarification not found in dataset."

    # Check if the continuation matches and determine the correct answer
    if ((continuation == row.iloc[0]["continuation of control sentence 1"] and sentence == row.iloc[0]["control sentence 1"]) 
        or (continuation == row.iloc[0]["continuation of control sentence 2"] and sentence == row.iloc[0]["control sentence 2"])):
            correct_answer = "yes"  
    else:
        correct_answer = "no"  

    # Compare the user's answer to the correct answer
    return f"Your answer '{agent_answer}' is correct" if agent_answer == correct_answer else f"Your answer '{agent_answer}' is incorrect"

    