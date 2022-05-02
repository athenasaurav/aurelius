import os, sys
import urllib.request
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify   
from werkzeug.utils import secure_filename 
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.retriever.dense import DensePassageRetriever
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.reader.farm import FARMReader
from haystack.pipeline import ExtractiveQAPipeline

if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    static_folder = os.path.join(sys._MEIPASS, 'static')
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
    # ui = FlaskUI(app, width=1920, height=1080)
else:
    app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'doc', 'docx'])

UPLOAD_FOLDER = '/root/aurelius'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def homeindex():
    return "OK"

@app.route('/index', methods=["POST"])
def index():
    # title = 'Prescient Automation calculator'
    global index_name, doc_store
    index_name = str(request.form.get('index_name').lower())
    doc_store = ElasticsearchDocumentStore(
        host='localhost',
        username='', password='',
        index=index_name
    )
    return 'Index Created'

@app.route('/update_index', methods=["POST"])
def update_index():
    # title = 'Prescient Automation calculator'
    global doc_store, index_name, filename
    index_name = str(request.form.get('index_name').lower())
    doc_store = ElasticsearchDocumentStore(
        host='localhost',
        username='', password='',
        index=index_name
    )
    if 'file' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp

    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message' : 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        resp = jsonify({'message' : 'File successfully uploaded'})
        resp.status_code = 201
        with open(filename, 'r') as f:
            data = f.read()
        data = data.split('\n')
        data_json = [
            {
                'text': paragraph,
                'meta': {
                    'source': 'meditations'
                }
            } for paragraph in data
        ]
        doc_store.write_documents(data_json)
        return resp
    else:
        resp = jsonify({'message' : 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'})
        resp.status_code = 400
        return resp
    


@app.route('/update_retriever', methods=["POST"])
def update_retriever():
    global index_name, doc_store, retriever
    index_name = str(request.form.get('index_name').lower())
    doc_store = ElasticsearchDocumentStore(
        host='localhost',
        username='', password='',
        index= index_name
    )
    retriever = DensePassageRetriever(
        document_store=doc_store,
        query_embedding_model='facebook/dpr-question_encoder-single-nq-base',
        passage_embedding_model='facebook/dpr-ctx_encoder-single-nq-base',
        use_gpu=False,
        embed_title=True
    )
    doc_store.update_embeddings(retriever=retriever)

    return 'Retriever Updated'


# initialize API
@app.route('/get_query', methods=["POST"])
def get_query(retriever_limit: int = 10, reader_limit: int = 3):
    """Makes query to doc store via Haystack pipeline.

    :param q: Query string representing the question being asked.
    :type q: str
    """
    # get answers
    global index_name, doc_store, retriever, pipeline
    index_name = str(request.form.get('index_name').lower())
    q = str(request.form.get('q').lower())
    DOC_STORE = ElasticsearchDocumentStore(
        host='localhost', 
        username='', 
        password='', 
        index=index_name
        )
    RETRIEVER = ElasticsearchRetriever(DOC_STORE)
    READER = FARMReader(model_name_or_path='deepset/bert-base-cased-squad2',
                        context_window_size=1500,
                        use_gpu=False)
    # initialize pipeline
    PIPELINE = ExtractiveQAPipeline(reader=READER, retriever=RETRIEVER)

    result = PIPELINE.run(query=q,
                        top_k_retriever=retriever_limit,
                        top_k_reader=reader_limit)
    return result





if __name__ == '__main__':
   app.run(debug = True, port = 8000, host = '0.0.0.0')
