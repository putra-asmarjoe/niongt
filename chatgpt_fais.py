import requests
import pandas as pd
import numpy as np
import faiss
import tiktoken
import sys

from openai import OpenAI
client = OpenAI()



model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = "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"

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}


def query(texts):
  response = requests.post(api_url,
                           headers=headers,
                           json={
                               "inputs": texts,
                               "options": {
                                   "wait_for_model": True
                               }
                           })
  return response.json()


texts = ["""
Title: The Adventures of Bleky the Brave
Once upon a time, in a quiet little town nestled between rolling hills
and a pristine river, there lived a dog named Bleky. Bleky was no
ordinary dog; he was known throughout the town for his extraordinary
bravery and unwavering loyalty.
"""]

output = query(texts)
# print("output : %s" % output)
# sys.exit();
embeddings = pd.DataFrame(output)
print(output)
print(embeddings)
embeddings.to_csv("data/embeddings.csv", index=False)


# Load embeddings from the CSV file
embeddings_df = pd.read_csv("data/embeddings.csv")
embeddings = embeddings_df.to_numpy().astype(np.float32)

# Initialize and populate a FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
    
def query_hf_model(text):
    response = requests.post(
        f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}",
        headers={"Authorization": f"Bearer {hf_token}"},
        json={"inputs": text, "options": {"wait_for_model": True}}
    )
    return response.json()
    
API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
headers = {"Authorization": "Bearer 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"}

def queryhugefc(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
    
# User query
user_query = input("Please enter your query: ")
user_query_embedding = np.array(query_hf_model([user_query])[0]).astype(np.float32)

user_queryExm = query(user_query)
# Find the most similar text in the dataset
_, indices = index.search(user_query_embedding.reshape(1, -1), 1)
most_similar_text_idx = indices[0][0]
most_similar_text = texts[most_similar_text_idx]

print("Query : %s" % user_query)
print("Most similiar text found: %s" % most_similar_text)


systemcontent = "You are a poetic assistant, skilled in explaining the answer based on the data provided. Do not seek answers outside the data that I have given. It is not permitted to answer with data outside the data that I have provided";
fulltextsubmit = systemcontent+most_similar_text+user_query;
numtoken = num_tokens_from_string(fulltextsubmit, "cl100k_base")
print("Number of tokens: %s" % numtoken)
# Query the OpenAI API


completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": systemcontent},
    {"role": "assistant", "content": most_similar_text},
    {"role": "user", "content": user_query}
  ]
)

output = queryhugefc({
        	"inputs": {
        		"question": user_query,
        		"context": most_similar_text
        	},
        })
print("Search by HF :", output)
print(completion)
print("Search by ChatGpt :", completion.choices[0].message.content)
