from flask import Flask, render_template, request, jsonify
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'docx'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  #16MB

model = Ollama(model="llama2", temperature=0.3)
embeddings = OllamaEmbeddings(model="llama2")
vectorstore = None

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

template = """You are a technical assistant. Provide answers with:
- Clear Markdown formatting
- Bullet points for lists
- Bold for key terms
- Section headers
- Line breaks between paragraphs

Context:
{context}

Question: {question}

Answer in this structured format:
**Summary** (1-2 sentences)
**Key Points** (bullet list)
**Additional Details** (if needed)"""

prompt = ChatPromptTemplate.from_template(template)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_and_process_documents(file_path, file_extension):
    try:
        if file_extension == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension == 'txt':
            loader = TextLoader(file_path)
        elif file_extension == 'docx':
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Unsupported file type")
        
        documents = loader.load()
        return text_splitter.split_documents(documents)
    except Exception as e:
        print(f"Document processing error: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'Allowed file types: PDF, TXT, DOCX'}), 400
    
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        file_extension = filename.rsplit('.', 1)[1].lower()
        chunks = load_and_process_documents(file_path, file_extension)
        
        global vectorstore
        if vectorstore is None:
            vectorstore = FAISS.from_documents(chunks, embeddings)
        else:
            vectorstore.add_documents(chunks)
        
        return jsonify({
            'status': 'success',
            'message': f'✅ {filename} processed successfully!'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'❌ Error: {str(e)}'
        }), 500

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    if not data:
        return jsonify({'answer': '❌ Invalid request format'}), 400
    
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'answer': '❌ Please enter a question'}), 400
    
    if vectorstore is None:
        return jsonify({'answer': 'ℹ️ Upload documents first to enable answers'}), 400
    
    try:
        relevant_docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        response = model.invoke(prompt.format(
            context=context,
            question=question
        ))
        
        formatted_response = response
        return jsonify({'answer': formatted_response})
    except Exception as e:
        return jsonify({'answer': f'⚠️ Error: {str(e)}'}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)