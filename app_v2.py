from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import Responses_from_listed_indexes
from Responses_from_listed_indexes import process_question, load_environment_variables, connect_to_index, MODEL, list_pinecone_indexes
from werkzeug.utils import secure_filename
import os
from Create_indexes import process_pdf


app = Flask(__name__, static_folder='static', static_url_path='')

load_environment_variables()
index = None

UPLOAD_FOLDER = 'uploaded_pdfs'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/styles.css')
def serve_static_css():
    return send_from_directory(app.static_folder, 'styles.css')


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdfFile' not in request.files:
        return jsonify({"error": "No file found."}), 400
    file = request.files['pdfFile']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Redirige a la ruta para ingresar el nombre del índice
        return redirect(url_for('enter_index_name', file_path=file_path))

    return jsonify({"error": "Invalid file format."}), 400


@app.route('/')
def index():
    indices = list_pinecone_indexes()
    return render_template('2index.html', indices=indices)


@app.route('/select_index', methods=['POST'])
def select_index():
    global index
    index_name = request.form['indexNameInput']  # Obtener el valor del input
    index = connect_to_index(index_name) 
    return jsonify({"result": "success"})



@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form['question']
    response = process_question(question, index, MODEL)
    return jsonify({"response": response})


@app.route('/enter_index_name/<path:file_path>', methods=['GET', 'POST'])
def enter_index_name(file_path):
    if request.method == 'POST':
        index_name = request.form['index_name']
        process_pdf(os.path.join(app.config['UPLOAD_FOLDER'], file_path), index_name)
        return redirect(url_for('index'))
    return render_template('enter_index_name.html', file_path=file_path)



if __name__ == '__main__':
    app.run(debug=True)



