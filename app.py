import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables (for local testing)
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Lecture Assistant", layout="wide")
st.title("🎓 AI Lecture Assistant")

# --- LANGCHAIN & DB SETUP ---
# Note: For deployment, you will set these as secrets in Streamlit Cloud
@st.cache_resource
def get_qa_chain():
    """Create and cache the QA chain to avoid reloading on every interaction."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    pinecone_vector_store = PineconeVectorStore.from_existing_index(
        index_name=os.environ.get("PINECONE_INDEX_NAME"),
        embedding=embeddings
    )
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.4)
    prompt_template = """
    You are an expert price action trader. Your task is to provide a comprehensive and detailed answer to the question based *only* on the provided context from lecture transcriptions, provide the context and all other relevant information.
    Synthesize all the relevant information from the context below to construct your answer.
    Use bullet points or numbered lists if it helps to structure the information clearly.
    If the answer is not found in the context, state that clearly. If you use any external knowledge state it clearly and focus on Al Brooks price action.


    CONTEXT:\n{context}\n\nQUESTION:\n{input}\n\nDETAILED ANSWER:
    """
    prompt = PromptTemplate.from_template(prompt_template)
    qa_chain = create_stuff_documents_chain(model, prompt)
    retriever = pinecone_vector_store.as_retriever(search_kwargs={"k": 7})
    return create_retrieval_chain(retriever, qa_chain)

chain = get_qa_chain()

# --- SESSION STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT INPUT ---
if prompt := st.chat_input("Ask a question about your lectures..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                response = chain.invoke({"input": prompt})
                full_response = response["answer"]
                message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"An error occurred: {e}"
                message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})