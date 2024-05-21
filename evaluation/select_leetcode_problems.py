import random
import csv
from datasets import load_dataset

dataset = load_dataset("greengerong/leetcode")

problem_data = dataset["train"]
problem_ids = problem_data["id"]
problem_slugs = problem_data["slug"]

selected_indices = random.sample(range(len(problem_ids)), min(100, len(problem_ids)))
selected_problems = [(problem_ids[i], problem_slugs[i]) for i in selected_indices]
csv_filename = 'selected_leetcode_problems.csv'
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(selected_problems)  # Write the selected problem IDs and slugs
