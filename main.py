from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# model = pipeline("text-generation","mistralai/Mistral-7B-Instruct-v0.1",device=0,max_length=256,truncation=True)
model = pipeline("text-generation", "distilgpt2", device=0, max_length=256, truncation=True)

llm = HuggingFacePipeline(pipeline=model)

template = PromptTemplate.from_template("Explain {topic} in detail for a {age} year old to understand.")

chain = template | llm

topic = input("Topic: ")
age = input("Age: ")

response = chain.invoke({"topic": topic, "age": age})

print(response)