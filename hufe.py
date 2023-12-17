import requests
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("question-answering", model="deepset/roberta-base-squad2");


API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": {
		"question": "What is my name?",
		"context": "My name is Clara and I live in Berkeley."
	},
})
# Mengekstrak hasil dari respons
# hasil = output.json()
print(output)
