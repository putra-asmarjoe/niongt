import os
from flask import Flask, request, jsonify
import tiktoken
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

app = Flask(__name__)
# Initialize Langchain models and Chroma
llm = OpenAI()
model_name = "gpt-3.5-turbo"
chat_model = ChatOpenAI(model_name=model_name)
chroma = Chroma()
model = SentenceTransformer('all-MiniLM-L6-v2')

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
        
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.split_documents(documents)
        
        
        # print('#########data save to chroma', documents)
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        # chroma.from_documents(file, embedding_function)
        # documents = text_splitter.split_documents(documents)
       
        vectordb = chroma.from_documents(all_splits, embedding_function)
        print(file)
    
    
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# Function to get relevant text from Chroma
def get_relevant_text(query, model, chroma, top_k=1):

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
    dataret = "\n\n######################".join(relevant_texts)
    
    print("*****************Chrome Response from database - relevant text:", dataret)
    print("==========Total Leng GPT relevant_text:", len(results))
    numtoken = num_tokens_from_string(dataret, "cl100k_base")
    print("==========Total token:", numtoken)

    return dataret


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


# Main script
if __name__ == "__main__":

    
    store_in_chroma(model, chroma);
    print("Welcome to the AI Assistant!")
    while True:
        user_text = input("Please enter your question (or 'x' to quit): ")
        if user_text.lower() == 'x':
            break

        relevant_text = get_relevant_text(user_text, model, chroma)
        # print("Chroma relevant_text:", relevant_text)
        combined_text = relevant_text + "\n\n" + user_text
        messages = [HumanMessage(content=combined_text)]

        
       
        # print("==========Response GPT relevant_text:", relevant_text)
        
        # response2 = llm.invoke(messages)
        # print("==========Response GPT LLMOpenAI:", response2)
        
        # response = chat_model.invoke(messages)
        # print("==========Response GPT ChatGpt3.5:", response)
        