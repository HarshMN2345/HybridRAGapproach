import os
from dotenv import load_dotenv
import streamlit as st
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from langchain_community.llms import Ollama

# Initialize the local Llama3.1 model
llm = Ollama(model="llama3.1")

# Neo4j connection details
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

# Initialize Neo4j driver
@st.cache_resource
def init_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Initialize sentence transformer model for semantic search
@st.cache_resource
def init_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_pdf_content():
    """Load the content of the PDF from the text file."""
    try:
        with open('extracted_text.txt', 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        st.error("Error: 'extracted_text.txt' file not found. Please ensure the file exists in the same directory as this script.")
        return None

@st.cache_data
def preprocess_and_index(_pdf_content):
    """Preprocess the PDF content, compute embeddings, and create a FAISS index."""
    content_chunks = _pdf_content.split('\n\n')  # Split into paragraphs
    
    # Compute embeddings
    model = init_sentence_transformer()
    embeddings = model.encode(content_chunks)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    return index, content_chunks

@st.cache_data
def load_or_create_index(_pdf_content):
    """Load existing index or create a new one if it doesn't exist."""
    return preprocess_and_index(_pdf_content)

def semantic_search(user_query, index, content_chunks, top_k=3):
    """Perform semantic search using FAISS."""
    model = init_sentence_transformer()
    query_vector = model.encode([user_query]).astype('float32')
    distances, indices = index.search(query_vector, top_k)
    return [content_chunks[i] for i in indices[0]]

def query_knowledge_graph(user_query):
    """Query the knowledge graph for relevant information."""
    driver = init_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (e:Entity)-[r:RELATED_TO]->(related:Entity)
            WHERE e.name CONTAINS $user_query OR related.name CONTAINS $user_query
            RETURN e.name AS entity, type(r) AS relation, related.name AS related_entity
            LIMIT 5
            """,
            user_query=user_query
        )
        return [f"{record['entity']} {record['relation']} {record['related_entity']}" 
                for record in result]

def generate_response(user_query, pdf_excerpts, kg_info):
    """Generate a response using the local Llama3.1 model based on the query, PDF excerpts, and knowledge graph info."""
    # Prepare context
    context = "\n".join(pdf_excerpts) + "\n" + "\n".join(kg_info)
    prompt = f"Context:\n{context}\n\nQuestion: {user_query}\n\nAnswer:"

    # Generate response using Llama3.1 model
    try:
        response = llm(prompt)
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    st.title("RAG Chatbot with Knowledge Graph")

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Load PDF content and create index (with loading indicator)
    with st.spinner("Initializing chatbot..."):
        pdf_content = load_pdf_content()
        if pdf_content is None:
            st.error("Failed to load PDF content. Please check the 'extracted_text.txt' file.")
            return
        index, content_chunks = load_or_create_index(pdf_content)

    # Chat interface
    user_input = st.text_input("You:", key="user_input")

    if user_input:
        with st.spinner("Generating response..."):
            relevant_excerpts = semantic_search(user_input, index, content_chunks)
            kg_info = query_knowledge_graph(user_input)
            response = generate_response(user_input, relevant_excerpts, kg_info)

        # Add user input and bot response to chat history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

    # Display chat history
    for role, message in st.session_state.chat_history:
        if role == "You":
            st.write(f"**You:** {message}")
        else:
            st.write(f"**Bot:** {message}")

if __name__ == "__main__":
    main()

