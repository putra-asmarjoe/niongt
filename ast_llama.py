from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.agent import OpenAIAssistantAgent
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

agent = OpenAIAssistantAgent.from_new(
    name="SEC Analyst",
    instructions="You are a QA assistant designed to analyze sec filings.",
    openai_tools=[{"type": "retrieval"}],
    instructions_prefix="Please address the user as Jerry.",
    files=["data/*"],
    verbose=True,
)

query = None
chat_history = []
while True:
  if not query:
    query = input("Prompt: ")
  if query in ['quit', 'q', 'exit']:
    sys.exit()
  response = agent.chat(query)
  print(str(response))

  chat_history.append((query, response))
  query = None
  #print(response)
