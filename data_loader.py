import pandas as pd
import random

def clean_data(data):
        """
        Cleans the dataset by removing " " and final stops.
        
        Args:
            data (pd.DataFrame): The dataset to clean.
        
        Returns:
            pd.DataFrame: The cleaned dataset.
        """
        data = data.map(lambda x: x.strip().rstrip('.').replace('"', '') if isinstance(x, str) else x)

        return data

class DataLoader:
    def __init__(self, csv_path):
        """
        Initializes the DataLoader with a dataset loaded from the given CSV file.
        
        Args:
            csv_path (str): Path to the CSV file containing the dataset.
        """
        self.data = clean_data(pd.read_csv(csv_path))
    
    def get_data(self):
        """
        Returns the loaded dataset as a pandas DataFrame.
        """
        return self.data

    def create_prompt(self, sentence_type, sentence, continuation):
        """
        Creates a prompt in the format:
        "{underspecified sentence or control sentence 1 or 2}. {continuation 1 or continuation 2}. Is the continuation correct?"
        
        Args:
            sentence_type (str): The type of the sentence ('underspecified' or 'control').
            sentence (str): The sentence (underspecified or control sentence 1/2).
            continuation (str): The continuation (continuation 1 or continuation 2).

        Returns:
            str: The formatted prompt.
        """
        # Validate the sentence type
        if sentence_type not in ["underspecified", "control"]:
            raise ValueError("Invalid sentence type. Must be 'underspecified' or 'control'.")

        # Check if the sentence exists in the dataset
        if sentence_type == "underspecified":
            row = self.data[self.data["underspecified sentence"] == sentence]
        else:
            row = self.data[
                (self.data["control sentence 1"] == sentence) | 
                (self.data["control sentence 2"] == sentence)
            ]
        
        if row.empty:
            raise ValueError("Sentence not found in dataset.")

        # Format the prompt
        return f"{sentence}. {continuation}. Is the continuation correct?"
    

    def generate_prompts(self, num_prompts=5):
        """
        Generates a specified number of unique prompts by sampling from the dataset
        without repetition within a single run.
        
        Args:
            num_prompts (int): The number of prompts to generate.
        
        Returns:
            list: A list of generated unique prompts.
        """
        # Ensure there are enough unique combinations in the dataset
        total_combinations = len(self.data) * 4  
        if num_prompts > total_combinations:
            raise ValueError("Requested more prompts than the total number of unique combinations available.")

        # Track used combinations to avoid repetition
        used_combinations = set()
        prompts = []

        while len(prompts) < num_prompts:
            # Randomly sample a row
            row_idx = random.randint(0, len(self.data) - 1)
            row = self.data.iloc[row_idx]

            # Randomly choose a sentence type
            sentence_type = random.choice(["underspecified", "control"])
            if sentence_type == "underspecified":
                sentence = row["underspecified sentence"]
            else:
                sentence = random.choice([row["control sentence 1"], row["control sentence 2"]])

            # Randomly choose a continuation
            continuation = random.choice([row["continuation of control sentence 1"], row["continuation of control sentence 2"]])

            # Create a unique identifier for the combination
            combination_key = (row_idx, sentence, continuation)

            # Skip if this combination has already been used
            if combination_key in used_combinations:
                continue

            # Add to used combinations and generate the prompt
            used_combinations.add(combination_key)
            
            prompt = f"{sentence}. {continuation}. Is the continuation correct?"
            prompts.append(prompt)

        return prompts


csv_path = "./data/experiment_2_data.csv"
data_loader = DataLoader(csv_path)

# Get the data
data = data_loader.get_data()

""""
# Generate some prompts
prompts = data_loader.generate_prompts(num_prompts=5)

with open("./data/prompts.txt", "w") as f:
    for prompt in prompts:
        f.write(prompt + "\n")

print("\nGenerated Prompts:")
for i, prompt in enumerate(prompts, start=1):
    print(f"{i}. {prompt}")
"""