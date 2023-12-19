import requests
import pandas as pd
import numpy as np
import faiss
import tiktoken
import os
import sys
import torch
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from openai import OpenAI
client = OpenAI()
from transformers import pipeline
checkpoint = "facebook/bart-base"
from transformers import BertModel, BertTokenizer

from langchain.llms import HuggingFaceHub
from getpass import getpass

model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = "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"

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

csv_file_path = "data/embeddings.csv";

llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.6})
# response  = llm.invoke("usd to idr ?")
# print(response)
# sys.exit(0);

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


# texts = ["""
# Title: The Adventures of Bleky the Brave
# Once upon a time, in a quiet little town nestled between rolling hills
# and a pristine river, there lived a dog named Bleky. Bleky was no
# ordinary dog; he was known throughout the town for his extraordinary
# bravery and unwavering loyalty.
# """]

# output = query(texts)
# embeddings = pd.DataFrame(output)
# print(output)
# print(embeddings)
# embeddings.to_csv("data/embeddings.csv", index=False)


feature_extractor = pipeline("feature-extraction",framework="pt",model=checkpoint)

systemcontent = "You are a modest-spirited assistant, adept in calculating, summarizing data, and creating statistics. Skilled in providing concise and clear explanations based on the provided data. You are not allowed to answer with data outside the data you have been given, unless you want to add additional information you possess that is relevant to the context of the question.";
       
texts = [];
# texts = ["""
# Title: The Adventures of Bleky the Brave
# Once upon a time, in a quiet little town nestled between rolling hills
# and a pristine river, there lived a dog named Bleky. Bleky was no
# ordinary dog; he was known throughout the town for his extraordinary
# bravery and unwavering loyalty.
# ""","""
# Bleky's owner, a kind-hearted woman named Emily, had raised him from a
# playful pup. From the very beginning, it was clear that Bleky was
# destined for greatness. His boundless energy and inquisitive nature made
# him the perfect companion for adventures.
# ""","""
# One sunny morning, as the golden rays of the sun painted the town in a
# warm glow, Bleky and Emily set out on one of their most memorable
# journeys. They decided to explore the dense forest that bordered the
# town, a place filled with mysteries and hidden wonders.
# """,]   
# texts = [
#     'Title: The Adventures of Bleky the Brave  \n \nOnce upon a time, in a quiet little town nestled between rolling hills \nand a pristine river, there lived a dog named Bleky. Bleky was no',
#     "ordinary dog; he was known throughout the town for his extraordinary \nbravery and unwavering loyalty.  \n \nBleky's owner, a kind-hearted woman named Emily, had raised him from a",
#     'playful pup. From the very beginning, it was clear that Bleky was \ndestined for greatness. His boundless energy and inquisitive nature made \nhim the perfect companion for adventures.',
#     'One sunny morning, as the golden rays of the sun painted the town in a \nwarm glow, Bleky and Emily set out on one of their most memorable',
#     'journeys. They decided to explore the dense forest that bordered the \ntown, a place filled with mysteries and hidden wonders.',
#     "As they ventured deeper into the forest, the sounds of chirping birds and \nrustling leaves surrounded them. Bleky's tail wagged enthusiastically,",
#     "and he led the way with his nose to the ground, following scents that \nonly he could detect.  \n \nIt wasn't long before they stumbled upon a distressed baby bird. The tiny",
#     'creature had fallen from its nest and was stranded on the forest floor. \nEmily gently cradled the baby bird in her hand, and Bleky watched with a \nconcerned expression.',
#     "With Bleky by her side, Emily carefully placed the baby bird back in its \nnest high above in the tree. Bleky's sharp eyes ensured they found the",
#     'right nest among the branches. The mother bird returned, chirping with \njoy, and they witnessed a heartwarming reunion. \n \nTheir adventure continued, and they encountered more challenges along the',
#     'way. Bleky fearlessly protected Emily from a curious snake and helped her \ncross a rushing stream by finding a safe path. His bravery knew no \nbounds.',
#     'As the day turned to evening, Bleky and Emily returned to their town, \ntired but content. The townsfolk gathered around to hear the tale of',
#     'their courageous adventure. Bleky received pats on the head and treats as \na token of appreciation for his bravery.  \n \nFrom that day forward, Bleky the Brave became a legend in the town. He',
#     'continued to embark on adventures with Emily, always ready to face any \nchallenge that came their way. The bond between Bleky and Emily grew',
#     'stronger with each passing day, a testament to the remarkable friendship \nbetween a dog and his owner.  \n \nAnd so, the adventures of Bleky the Brave continued, filling the town',
#     'with stories of courage, loyalty, and the unbreakable bond between a dog \nand his beloved owner.'
# ]


def store_in_csv():
    documents = []
    for file in os.listdir('data'):
        print("Scanning files... ",file )
        if file.endswith('.pdf'):
          pdf_path = 'data/' + file
          loader = PyPDFLoader(pdf_path)
          documents.extend(loader.load())
          # elif file.endswith('.docx') or file.endswith('.doc'):
          #     doc_path = 'data/' + file
          #     loader = Docx2txtLoader(doc_path)
          #     documents.extend(loader.load())
          # elif file.endswith('.txt'):
          # text_path = 'data/' + file
          # loader = TextLoader(text_path)
          # documents.extend(loader.load())
          
      
          # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
          # documents = text_splitter.split_documents(documents)
          
          
          documents = loader.load()
  
          # split it into chunks
          # text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
          # docs = text_splitter.split_documents(documents)
          text_splitter = RecursiveCharacterTextSplitter(
            #   separators=["\n\n", "\r\n","\n", " "],
              chunk_size=200, 
              chunk_overlap=0, 
              add_start_index=True
          )
          all_splits = text_splitter.split_documents(documents)
          
          for item in all_splits:
            textfex = item.page_content;
            texts.append(textfex)
            
          
        #   print(texts)  
        #   print(len(texts))  
            
          
          output = query(texts)
          # print("output : %s" % output)
          # sys.exit();
          embeddings = pd.DataFrame(output)
          # print(output)
          # print(embeddings)
          embeddings.to_csv("data/embeddings.csv", index=False)  
          
          # create the open-source embedding function
        #   embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
          # for item in all_splits:
          #   textfex = item.page_content;
          #   #Reducing along the first dimension to get a 768 dimensional array
          #   textfexextra = feature_extractor(textfex,return_tensors = "pt")
          #   print(textfexextra)
            
          #   # Convert to numpy array
          #   last_hidden_states_np = textfexextra.squeeze().numpy()
          #   print(last_hidden_states_np)
            
          #   df = pd.DataFrame(last_hidden_states_np)
            
          #   # df.to_csv("data/embeddings.csv", index=False)
            
            
          #    # If the file doesn't exist, write a new file with headers
          #   df.to_csv(csv_file_path, mode='w', header=True, index=False)
            # print(item.page_content)
            # len(item.page_content)
          
          # load it into Chroma
          # chroma.from_documents(all_splits, embedding_function)
        
     
store_in_csv();
# sys.exit();


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
    
# Main script
if __name__ == "__main__":

    print("Welcome to the AI Assistant!")
    while True:
        user_text = input("Please enter your question (or 'x' to quit): ")
        if user_text.lower() == 'x':
            break
        user_query = user_text;#"dog name"

       
        
        
        
        
        
        # Load embeddings from CSV
        # df = pd.read_csv(csv_file_path)
        # embeddings = np.array(df.values, dtype='float32')  # Ensure the data type is float32
        
        # Create a FAISS index
        # dimension = embeddings.shape[1]  # Dimension of the embeddings
        # index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity
        # index.add(embeddings)  # Add embeddings to the index
        
        # Process user query
        # Assuming 'query_embedding' is the embedding vector for the user query
        query_embedding = query(user_query)  # Convert user query to embedding
        query_embedding = np.array(query_embedding, dtype='float32').reshape(1, -1)
        
        # Perform similarity search
        k = 5  # Number of nearest neighbors to find
        D, I = index.search(query_embedding, k)  # D is distance, I is index
        
        # Retrieve relevant texts
        # Assuming 'texts' is a list or a pandas Series of the original texts
        most_similar_text2  = "";
        data_countFaiss = len(I[0])
        for i in I[0]:
            most_similar_text2 += "" + texts[i]
            print(texts[i])
            
            
        # output2 = queryhugefc({
        #         	"inputs": {
        #         		"question": user_query,
        #         		"context": most_similar_text2
        #         	},
        #         })
        # print("Search by HF :", output2)    
        
        
        print("Total Faiss Similiar Text Faiss ",data_countFaiss)
        print("\n\n")  
        
        fulltextsubmit = systemcontent+most_similar_text2+user_query;
        numtoken = num_tokens_from_string(fulltextsubmit, "cl100k_base")
        print("Number of tokens: %s" % numtoken)    
            
            
        # response  = llm(most_similar_text2 +", and questions is %s" % user_query)
        response  = llm(most_similar_text2 +", and questions is %s" % user_query)
        print("Search by hfLLM :", response)   
            
        print("\n\n")  
        completion = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
            {"role": "system", "content": systemcontent},
            {"role": "assistant", "content": most_similar_text2},
            {"role": "user", "content": user_query}
          ],
            max_tokens=50,
            temperature=0.6
        )
        print("Token by ChatGpt :", completion.usage)
        print("Search by ChatGpt:", completion.choices[0].message.content)
