import os
import torch
import numpy as np
import yaml
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from crewai import Agent, Task, Crew
from crewai_tools import FileReadTool

# Initialize the tool to read any files the agents knows or lean the path for
file_read_tool = FileReadTool()

# Initialize the tool with a specific file path, so the agent can only read the content of the specified file
file_read_tool = FileReadTool(file_path='path/to/your/file.txt')

# Load pre-trained model and tokenizer from Hugging Face
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Define a function to generate text using the Hugging Face model
def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Define the agent's role, goal, and backstory
role1 = "You are a logistics coordinator responsible for matching clients with suppliers based on their product needs and availability."
goal1 = "Your goal is to ensure that clients receive the necessary products in a timely manner, especially during natural disasters."
backstory1 = "In the wake of recent natural disasters, there has been a significant disruption in supply chains. Your task is to streamline the process of matching clients who need products with suppliers who can provide them, thereby alleviating distress and saving lives."

# Create the first agent with the new text generation function
agent1 = Agent(
    role=role1,
    goal=goal1,
    backstory=backstory1,
    verbose=False,
    tools=[file_read_tool],
    generate_text=generate_text  # Add the text generation function to the agent
)

# Define the second agent's role, goal, and backstory
role2 = "You are a supplier coordinator responsible for finding suppliers for the products identified by the logistics coordinator."
goal2 = "Your goal is to ensure that the identified products are matched with the best available suppliers."
backstory2 = "In the wake of recent natural disasters, there has been a significant disruption in supply chains. Your task is to find suppliers who can provide the necessary products identified by the logistics coordinator."

# Create the second agent
agent2 = Agent(
    role=role2,
    goal=goal2,
    backstory=backstory2,
    verbose=False,
    tools=[file_read_tool]
)

task1 = Task(
    description="Given the user´s query, match them with the most suitable product",
    expected_output="A list of products that match the user´s query, in order of relevance",
    agent=agent1
)

# Define a new task that takes the user's input and matches it to the best option in the document
task2 = Task(
    description="Given a description of an item, find the best match in the document",
    expected_output="The best matching item from the document",
    agent=agent1
)

# Define a task for the second agent to match the product with suppliers
task3 = Task(
    description="Match the identified product with the best available suppliers",
    expected_output="A list of suppliers that can provide the identified product",
    agent=agent2
)

# Create a crew that operates in a linear way
crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2, task3],  # Add the tasks in a linear sequence
    verbose=True
)

# Function to accept user input and process it
def process_user_input(user_input):
    # Create a prompt for the agent based on the user's input
    prompt = f"Find the best match for the following item description: {user_input}"
    # Use the agent's generate_text function to get the response
    response = agent1.generate_text(prompt)
    return response

# Function to match the product with suppliers
def match_with_suppliers(product, suppliers):
    # Find suppliers that have the product
    matching_suppliers = [supplier for supplier, products in suppliers.items() if product in products]
    return matching_suppliers

# Load suppliers from YAML file
def load_suppliers(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data['suppliers']

# Start the crew
crew.kickoff()

# Example usage
user_input = "A high-quality, durable tent for camping in extreme weather conditions"
product = process_user_input(user_input)
print(f"Product: {product}")

# Load suppliers from the YAML file
suppliers = load_suppliers('suppliers.yml')
matching_suppliers = match_with_suppliers(product, suppliers)
print(f"Suppliers: {matching_suppliers}")