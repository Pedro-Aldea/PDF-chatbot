from flask import Flask, render_template, request, jsonify
import Responses_from_listed_indexes
from Responses_from_listed_indexes import process_question, load_environment_variables, connect_to_index, MODEL, list_pinecone_indexes

app = Flask(__name__)

load_environment_variables()
index = None

@app.route('/')
def index():
    indices = list_pinecone_indexes()
    return render_template('index.html', indices=indices)

@app.route('/select_index', methods=['POST'])
def select_index():
    global index
    index_name = request.form['indexName']
    index = connect_to_index(index_name) 
    return jsonify({"result": "success"})

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form['question']
    response = process_question(question, index, MODEL)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)



