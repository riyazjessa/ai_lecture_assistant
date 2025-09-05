import os
import datetime
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables from .env file
load_dotenv()

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")

# --- LANGCHAIN & DB SETUP (Done once on startup) ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
pinecone_vector_store = PineconeVectorStore.from_existing_index(
    index_name=os.environ.get("PINECONE_INDEX_NAME"),
    embedding=embeddings
)

def create_qa_chain():
    """Creates the question-answering chain."""
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.4)
    prompt_template = """
    You are an expert price action trader. Your task is to provide a comprehensive and detailed answer to the question based *only* on the provided context from lecture transcriptions, provide the context and all other relevant information.
    Synthesize all the relevant information from the context below to construct your answer.
    Use bullet points or numbered lists if it helps to structure the information clearly.
    If the answer is not found in the context, state that clearly. If you use any external knowledge state it clearly and focus on Al Brooks price action.


    CONTEXT:
    {context}

    QUESTION:
    {input}

    DETAILED ANSWER:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    qa_chain = create_stuff_documents_chain(model, prompt)
    retriever = pinecone_vector_store.as_retriever(search_kwargs={"k": 7})
    return create_retrieval_chain(retriever, qa_chain)

qa_chain = create_qa_chain()

# --- WEB ROUTES ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == os.environ.get("APP_USERNAME") and password == os.environ.get("APP_PASSWORD"):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return "Invalid credentials", 401
    return render_template('login.html')

@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/ask', methods=['POST'])
def ask():
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_query = request.json.get('question')
    if not user_query:
        return jsonify({"error": "No question provided"}), 400

    # --- LOGGING USAGE ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"LOG: User='{session['username']}', Timestamp='{timestamp}', Query='{user_query}'")
    # This print statement will appear in Render's log stream.

    try:
        response = qa_chain.invoke({"input": user_query})
        return jsonify({"answer": response["answer"]})
    except Exception as e:
        print(f"ERROR: {e}")
        return jsonify({"error": "An error occurred while generating the answer."}), 500

if __name__ == '__main__':
    # This runs the app locally for testing
    app.run(debug=True)