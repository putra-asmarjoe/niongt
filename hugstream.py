import requests
import pandas as pd
import numpy as np
import faiss
import tiktoken
from dotenv import load_dotenv
import os
import sys
import torch
import pdfplumber
import pandas as pd
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from openai import OpenAI

# Memuat variabel dari .env
load_dotenv()

client = OpenAI()
from transformers import pipeline
checkpoint = "facebook/bart-base"
from transformers import BertModel, BertTokenizer

from langchain.llms import HuggingFaceHub

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = os.getenv('API_KEY')
sapa_token = os.getenv('SAPADATA_CHAT_TOKEN')

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": "Bearer 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"}

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
 

feature_extractor = pipeline("feature-extraction",framework="pt",model=checkpoint)

systemcontent = "You are a modest-spirited assistant, adept in calculating, summarizing data, and creating statistics. Skilled in providing concise and clear explanations based on the provided data. You are not allowed to answer with data outside the data you have been given, unless you want to add additional information you possess that is relevant to the context of the question.";
       
texts = [];  

def extract_tables_from_pdf(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Mengekstrak tabel dari halaman
            page_tables = page.extract_tables()
            for table in page_tables:
                # Mengonversi tabel ke DataFrame
                df = pd.DataFrame(table[1:], columns=table[0])
                tables.append(df)
    return tables

# pdf_path = 'data/data.pdf'
# extracted_tables = extract_tables_from_pdf(pdf_path)

# # Contoh untuk menampilkan tabel pertama yang diekstrak
# if extracted_tables:
#     print(extracted_tables[0])
    
# sys.exit(0);


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
              chunk_overlap=30, 
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
    
    
# Fungsi untuk menampilkan teks di bagian bawah
def show_text_at_bottom(text):
    st.empty()  # Buat area kosong di bawah
    with st.container():  # Gunakan container untuk menempatkan teks di tengah
        st.write(text)  # Tampilkan teks
   
   
def show_result_query(user_query):
        
        print("Search in Faiss : ",user_query)
        # Store embeddings from the CSV file     
        store_in_csv();
        # Load embeddings from the CSV file
        embeddings_df = pd.read_csv("data/embeddings.csv")
        embeddings = embeddings_df.to_numpy().astype(np.float32)
        
        # Initialize and populate a FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
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
            
        
        # Perform similarity search
        k2 = 10  # Number of nearest neighbors to find
        D, I2 = index.search(query_embedding, k2)  # D is distance, I is index
        
        # Retrieve relevant texts
        # Assuming 'texts' is a list or a pandas Series of the original texts
        most_similar_text3  = "";
        data_countFaiss = len(I2[0])
        for i in I2[0]:
            most_similar_text3 += "" + texts[i]
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
            
        
        
        ttlgpttoken = "";
        gptresponds = "";
            
        # response  = llm(most_similar_text2 +", and questions is %s" % user_query)
        response  = llm(most_similar_text3 +", and questions is %s" % user_query)
        print("Search by hfLLM :", response)
        completion = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
            {"role": "system", "content": systemcontent},
            {"role": "assistant", "content": most_similar_text2},
            {"role": "user", "content": user_query}
          ],
            max_tokens=150,
            temperature=0.6
        )
        
        
        ttlgpttoken = completion.usage;
        gptresponds = completion.choices[0].message.content;
        print("Token by ChatGpt :", ttlgpttoken)
        print("Search by ChatGpt:", gptresponds)
        
        with st.container():  # Gunakan container untuk menempatkan teks di tengah 
            st.info("Faiss Similiar Text: %s" % (most_similar_text2), icon="🗺")  # Tampilkan teks    
            st.info("Faiss tokens: %s \n\nGpt tokens: %s" % (numtoken, ttlgpttoken), icon="🗺")  # Tampilkan teks   
        
        st.session_state.messages.append({"role": "ai", "content": "Faiss Similiar Text: %s \n\n Faiss tokens: %s \n\nGpt tokens: %s" %(most_similar_text2, numtoken, ttlgpttoken)})
                
        
        st.empty()  # Buat area kosong di bawah
        # with st.container():  # Gunakan container untuk menempatkan teks di tengah
        #     st.success("Answer by hfLLM : %s"% response, icon="👾")  # Tampilkan teks
        
        st.session_state.messages.append({"role": "assistant", "content": "hfLLM : %s \n\nGpt : %s "% (response, gptresponds)})
        st.chat_message("assistant").write("hfLLM : %s \n\nGpt : %s "% (response, gptresponds))

def get_pdf_text(pdf_docs):
    

    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
    
def loadfolder_pdf_text():
    
    text = ""
    for file in os.listdir('data'):
        print("Scanning files... ",file )
        if file.endswith('.pdf'):
            pdf_path = 'data/' + file
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text()
    return text;


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
 
def get_conversation_chain_openai(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True) 
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    print(memory)
    return conversation_chain

def get_conversation_chain_openai2(vectorstore):
     # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True) 
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    print(conversation_chain)
    return conversation_chain

def get_conversation_chain_mixtrail(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    print("conversation_chain = " + str(conversation_chain))
    return conversation_chain
    
def router_conversation_chain(model_name, vectorstore):
    conversation_chain = None;
    if "GPT-3.5" in model_name:
        conversation_chain = get_conversation_chain_openai(vectorstore)
    elif "mixtral" in model_name.lower():
        conversation_chain = get_conversation_chain_mixtrail(vectorstore)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return conversation_chain;
  
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    print("#START1")
    print(st.session_state.conversation)
    print("#START")
    print(st.session_state.chat_history)
    print("#END")
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:        
            st.chat_message("user").write("%s "% (message.content))
        else:            
            st.chat_message("assistant").write("%s "% (message.content))

# # Main script
# if __name__ == "__main__":
def main():  
    # print("APLIKASI START RUN................................................................")
    
    # #READ PDF 
    
    # # get pdf text
    # raw_text = loadfolder_pdf_text();     

    # # get the text chunks
    # text_chunks = get_text_chunks(raw_text)

    # # create vector store
    # vectorstore = get_vectorstore(text_chunks)

    # # create conversation chain
    # #get_conversation_chain_openai
    # conversation_chain = get_conversation_chain_openai2(vectorstore) 
    
    # while True:
    #         user_text = input("Please enter your question (or 'x' to quit): ")
    #         if user_text.lower() == 'x':
    #             break
                
            
    #         memorytext = conversation_chain(user_text);
    #         print(memorytext)
    #         memorytextchat = memorytext['chat_history']
    #         print(memorytextchat)
    #   if "conversation" not in st.session_state:
    #         st.session_state.conversation = None
    #   if "chat_history" not in st.session_state:
    #         st.session_state.chat_history = None
      
    
      # Judul Aplikasi
      st.title("🤖 SapaData 🧠")
    #   st.caption("Where Data Finds Its Voice, Transforming Data into Dialogues")
      st.caption("Transforming Data Voice into Dialogues")
     
      model = st.radio(
        "",
            options=["⛰️ Mixtral","✨ GPT-3.5"],
            index=0,
            horizontal=True,
        )
        
      st.session_state["model"] = model
      
      if "model" not in st.session_state:
        st.session_state["model"] = model
        
         
      
        
      container = st.container(border=True)
      with st.sidebar:
          sapa_key = st.text_input("SapaData Key", key="chatbot_api_key", type="password", value="")          
          "[Hub kami untuk mendapatkan SapaData Key .."
          st.subheader("Your documents")
          pdf_docs = st.file_uploader(
                "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
          
            
          if st.button("Process"):
            #save to
            st.session_state["docname"] = len(pdf_docs)
            
            with st.spinner("Processing"):
                # container.success("Anda telah mengupload file %s , Mulai chat untuk berbicara dengan data and"% (pdf_docs[0].name))
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = router_conversation_chain(st.session_state["model"], vectorstore)
                    
      if "messages" not in st.session_state:
          st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
      
      if sapa_key == "":
        st.session_state["messages"] = [{"role": "assistant", "content": "Sebelum memulai, Masukan SapaData Key Anda Pada Panel Kiri .."}]
      else:
        if(sapa_token != sapa_key):
            st.session_state["messages"] = [{"role": "assistant", "content": "SapaData Key Anda Tidak Valid, Silahkan periksa ulang Key Anda"}]
        else:            
            
            if "docname" not in st.session_state:        
              st.session_state["messages"] = [{"role": "assistant", "content": "Halo dokumen apa yang perlu kita sapa??"}]
            else:
              st.session_state["messages"] = [{"role": "assistant", "content": "Anda telah mengupload %s file, Mulai chat untuk berbicara dengan data anda"% (st.session_state["docname"])}]
      
      
      for msg in st.session_state.messages:
          st.chat_message(msg["role"]).write(msg["content"])
      
      
    
      # Inisialisasi session_state untuk menyimpan teks
      if 'input_text' not in st.session_state:
          st.session_state['input_text'] = ''
      
      
      # # Logika untuk menampilkan teks ketika tombol diklik
      if prompt := st.chat_input():
          st.session_state['input_text'] = prompt
          if sapa_key == "":
            st.info("Silahkan Masukan SapaData Key Anda, Belum punya SapaData Key? Hub kami disini .. ", icon="ℹ️")
          else:
            if(sapa_token != sapa_key):
                st.info("SapaData Key Anda Tidak Valid", icon="ℹ️")
            else:
                if "docname" not in st.session_state:        
                    st.session_state["messages"] = [{"role": "assistant", "content": "Silahkan upload dokumen anda terlebih dahulu ??"}]
                else:
                    st.session_state['input_text'] = prompt
                    handle_userinput(prompt)
      
      
                # st.write("You entered:", openai_api_key)  
                
    
    
        
           
if __name__ == '__main__':
    main()