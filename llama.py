from flask import Flask, request, jsonify
from llama_index import VectorStoreIndex, SimpleDirectoryReader

app = Flask(__name__)

# Function to initialize or reinitialize the index
def initialize_index():
    documents = SimpleDirectoryReader("data").load_data()
    return VectorStoreIndex.from_documents(documents)

# Initialize the index at the start
index = initialize_index()
query_engine = index.as_query_engine()

@app.route('/query', methods=['POST'])
def handle_query():
    query = request.form.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        response = query_engine.query(query)
        if not isinstance(response, str):
            response = str(response)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/refreshdata', methods=['GET', 'POST'])
def refresh_data():
    try:
        # Reinitialize the index
        global index, query_engine
        index = initialize_index()
        query_engine = index.as_query_engine()
        return jsonify({"success": "Data reindexed successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)