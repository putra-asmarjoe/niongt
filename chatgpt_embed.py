import os
from flask import Flask, request, jsonify
import tiktoken
import requests
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings



# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("question-answering", model="deepset/roberta-base-squad2");


app = Flask(__name__)
# Initialize Langchain models and Chroma
llm = OpenAI()
model_name = "gpt-3.5-turbo"
chat_model = ChatOpenAI(model_name=model_name)
chroma = Chroma()
model = SentenceTransformer('all-MiniLM-L6-v2')


# Initialize the embeddings module from LangChain
openai_embeddings = OpenAIEmbeddings()
# Function to store vectors in Chroma
def store_vectors_in_chroma(texts, model, chroma):
    embeddings = model.encode(texts, convert_to_tensor=False)
    # Convert embeddings to a list of NumPy arrays
    embeddings_list = [embedding.tolist() for embedding in embeddings]
    
    # Use the text content as the ID
    text_ids = [text[:50] for text in texts]  # You can adjust the length as needed
    
    # Store texts, embeddings, and metadata in Chroma
    chroma.add_texts(text_ids, texts, embeddings_list, metadata_list=[])
    
def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs
  
def store_in_chroma(model, chroma):
    documents = []
    for file in os.listdir('data'):
        # if file.endswith('.pdf'):
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
        
        print(file)
    
        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
        # documents = text_splitter.split_documents(documents)
        
        
        documents = loader.load()

        # split it into chunks
        # text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        # docs = text_splitter.split_documents(documents)
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\r\n","\n", " "],
            chunk_size=200, 
            chunk_overlap=0, 
            add_start_index=True
        )
        all_splits = text_splitter.split_documents(documents)
        
        
        # create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # load it into Chroma
        chroma.from_documents(all_splits, embedding_function)
    
    
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# Function to get relevant text from Chroma
def get_relevant_text(query, model, chroma, top_k=5):

    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # load it into Chroma
    query_vector = model.encode(query, convert_to_tensor=False).tolist()  # Ensure it's in the correct format
    results = chroma.similarity_search_by_vector(query_vector, top_k=top_k)
    
    # db = Chroma.from_documents(docs, embedding_function)
    # docs = db.similarity_search(query)
    # # print results
    # print(docs[0].page_content)

    # print("Chrome Response from Chrom database:", results)
    # relevant_texts = [result.metadata['text'] for result in results if 'metadata' in result and 'text' in result['metadata']]
    relevant_texts = [result.page_content for result in results]
    dataret = " ".join(relevant_texts)
    
    # print("*****************Chrome Response from database - relevant text:", dataret)
    print("==========Total Len relevant_text:", len(results))
    numtoken = num_tokens_from_string(dataret, "cl100k_base")
    print("==========Total get_relevant_text numtoken:", numtoken)

    return dataret


API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
# Function to delete data vectors in Chroma
def delete_chroma_data(chroma, collection_name):
    chroma.delete_collection(collection_name)
# Initialize DirectoryLoader
directory_loader = DirectoryLoader('data')

# store_in_chroma(model, chroma);
# @app.route('/query', methods=['POST'])
# def api_ask():
#     user_text = request.form.get('query')
#     if not user_text:
#         return jsonify({"error": "No query provided"}), 400

    
#     relevant_text = get_relevant_text(user_text, model, chroma)
#     combined_text = relevant_text + "\n\n" + user_text
#     messages = [HumanMessage(content=combined_text)]
    
#     try:
#         response2 = llm.invoke(messages)
#         if not isinstance(response, str):
#             response = str(response)
#         return jsonify({"response": response2})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # ... (add more endpoints as needed)

# if __name__ == '__main__':
#     app.run(debug=True, port=5001)

# idsave = 19023;
# texttoembed = "bleky friend name is Jero,jero is cat";
# response = client.embeddings.create(
#     input=texttoembed,
#     model="text-embedding-ada-002"
# )
# dataret = response.data[0].embedding;
# print(dataret)
# numtoken = num_tokens_from_string(texttoembed, "cl100k_base")
# print("==========Text embed:", texttoembed)
# print("==========Total token embed:", numtoken)

# query_result = openai_embeddings.embed_query(texttoembed)
# print("==========Total LC EMbed:", query_result)

# load the document and split it into chunks
# loader = TextLoader("data/data.txt")
# documents = loader.load()

# # split it into chunks
# # text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
# # docs = text_splitter.split_documents(documents)
# text_splitter = RecursiveCharacterTextSplitter(
#     separators=["\n\n", "\r\n","\n", " "],
#     chunk_size=200, 
#     chunk_overlap=0, 
#     add_start_index=True
# )
# all_splits = text_splitter.split_documents(documents)

# docstxt = text_splitter.split_text(texttoembed)
# len(docstxt)

# # create the open-source embedding function
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# # load it into Chroma
# db = Chroma.from_documents(all_splits, embedding_function)

store_in_chroma(model, chroma)
# Main script
if __name__ == "__main__":

    print("Welcome to the AI Assistant!")
    while True:
        user_text = input("Please enter your question (or 'x' to quit): ")
        if user_text.lower() == 'x':
            break
        querysrc = user_text;#"dog name"
        query_vector = model.encode(querysrc, convert_to_tensor=False).tolist()  # Ensure it's in the correct format
        docssimiliar = chroma.similarity_search_by_vector(query_vector,5)
        
        # print results
        print("docssimiliar", docssimiliar)
        
        relevant_texts = [result.page_content for result in docssimiliar]
        dataretStr = " ".join(relevant_texts)
        numtoken = num_tokens_from_string(dataretStr, "cl100k_base")
        print("==========Total Len docssimiliar:", len(docssimiliar))
        print("==========Text docssimiliar numtoken:", numtoken)
        
        print("==========def relevant_text:")        
        relevant_text = get_relevant_text(user_text, model, chroma)
        
        
        
        output = query({
                	"inputs": {
                		"question": querysrc,
                		"context": dataretStr
                	},
                })
        print("Search by HF", output)
        output = query({
                	"inputs": {
                		"question": querysrc,
                		"context": relevant_text
                	},
                })
        print("Search by HF2", output)
        
        # combined_text = relevant_text + "\n\n" + querysrc
        # response2 = llm.invoke(combined_text)
        # print("==========Response from GPT LLM:", response2)
        
        # messagesStr = HumanMessage(content=combined_text)
        # messages = [messagesStr]
        # response = chat_model.invoke(messages)
        # print("==========Response from GPT Chat:", response)
        
        # Mengekstrak hasil dari respons
#     store_in_chroma(model, chroma);
#     print("Welcome to the AI Assistant!")
#     while True:
#         user_text = input("Please enter your question (or 'x' to quit): ")
#         if user_text.lower() == 'x':
#             break

#         relevant_text = get_relevant_text(user_text, model, chroma)
#         # print("Chroma relevant_text:", relevant_text)
#         combined_text = relevant_text + "\n\n" + user_text
#         messagesStr = HumanMessage(content=combined_text)
#         messages = [messagesStr]

         
        	
#         output = query({
#         	"inputs": {
#         		"question": user_text,
#         		"context": combined_text
#         	},
#         })
#         # Mengekstrak hasil dari respons
#         print(output)
       
#         # print("==========Response GPT relevant_text:", relevant_text)
        
#         # response2 = llm.invoke(messages)
#         # print("==========Response GPT LLMOpenAI:", response2)
        
#         # response = chat_model.invoke(messages)
#         # print("==========Response GPT ChatGpt3.5:", response)
        