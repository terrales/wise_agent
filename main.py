from graph_builder import *
from memory import *
#from retriever import retriever
from utils import _print_event, add_to_results
import uuid
import os
import getpass
from data_loader import *
import json

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# Prompt for API keys if not already set
os.environ["MISTRAL_API_KEY"] = ""

#TODO

def main():
    graph = build_graph()
    memory = Memory()

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    _printed = set()
    results = []

    data_loader = DataLoader("./data/experiment_2_data.csv")
    prompts = data_loader.generate_prompts(num_prompts=8)

    with open("./data/prompts.txt", "w") as f:
        for prompt in prompts:
            f.write(prompt + "\n")
    for prompt in prompts:

        result = {}
        events = graph.stream({"messages": ("user", prompt)}, config, stream_mode="values")
        for event in events:
            _print_event(event, _printed)
            add_to_results(event, result)
        
        results.append(result)

    with open("./data/results.txt", "w")as f:
        for result in results:
            f.write(str(result) + "\n")
       # json.dump(results, f)

    save_to_file(memory, "./data/memory.json")

""""
    while True:
        user_input = input("Your message: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        result = {}
        events = graph.stream({"messages": ("user", user_input)}, config, stream_mode="values")
        for event in events:
            _print_event(event, _printed)
            add_to_results(event, result)
        
        results.append(result)
"""


if __name__ == "__main__":
    main()
