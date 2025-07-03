import streamlit as st
import google.generativeai as genai
import os
import time # Import time
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np
from sentence_transformers import SentenceTransformer
import torch  # Import torch
from serpapi.google_search import GoogleSearch
from dotenv import load_dotenv # Import dotenv
from gmail_reader import GmailReader # Import GmailReader
from email_query_processor import EmailQueryProcessor # Import EmailQueryProcessor

load_dotenv() # Load environment variables from .env file

SYSTEM_PROMPT_FILE = "system_prompt.txt"

def load_system_prompt():
    if os.path.exists(SYSTEM_PROMPT_FILE):
        with open(SYSTEM_PROMPT_FILE, "r") as f:
            return f.read()
    return ""

def save_system_prompt(prompt):
    with open(SYSTEM_PROMPT_FILE, "w") as f:
        f.write(prompt)

# Replace 'YOUR_API_KEY' with your actual Gemini API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
GOOGLE_APP_PASSWORD = os.environ.get("GOOGLE_APP_PASSWORD") # Load Google App Password
GOOGLE_EMAIL_ADDRESS = os.environ.get("GOOGLE_EMAIL_ADDRESS") # Load Google Email Address

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize Qdrant client
QDRANT_HOST = "https://741d8e2a-6524-4f1e-a9e0-0f7021ef7d51.us-east4-0.gcp.cloud.qdrant.io"
QDRANT_PORT = None # Port is part of the host URL

client = QdrantClient(
    url=QDRANT_HOST,
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY,
)

COLLECTION_NAME = "gemini_chat_history"
VECTOR_SIZE = 384  # Size of the Sentence Transformer embeddings

# Initialize Qdrant collection if it doesn't exist
try:
    client.get_collection(collection_name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' already exists.")
except:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' created.")

# Load Sentence Transformer model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print(next(embedding_model.parameters()).device)

def generate_embedding(text):
    """Generate embedding for the given text using Sentence Transformers."""
    return embedding_model.encode(text).tolist()

def perform_web_search(query):
    """Performs a web search using SerpAPI and returns formatted results."""
    if not SERPAPI_API_KEY or SERPAPI_API_KEY == "YOUR_SERPAPI_API_KEY":
        st.warning("SerpAPI key not set. Web search will not function.")
        return "Web search is not configured."

    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "organic_results" in results:
            formatted_results = "Search Results:\n"
            for i, res in enumerate(results["organic_results"][:5]): # Limit to top 5 results
                formatted_results += f"{i+1}. Title: {res.get('title', 'N/A')}\n"
                formatted_results += f"   Link: {res.get('link', 'N/A')}\n"
                formatted_results += f"   Snippet: {res.get('snippet', 'N/A')}\n\n"
            return formatted_results
        elif "answer_box" in results:
            return f"Answer: {results['answer_box'].get('answer', results['answer_box'].get('snippet', 'N/A'))}\n"
        elif "sports_results" in results:
            return f"Sports Results: {results['sports_results'].get('title', 'N/A')}\n"
        elif "knowledge_graph" in results:
            return f"Knowledge Graph: {results['knowledge_graph'].get('description', 'N/A')}\n"
        else:
            return "No relevant search results found."
    except Exception as e:
        return f"An error occurred during web search: {e}"

def should_perform_web_search(query):
    """Determines if a web search is needed based on the query."""
    # Simple heuristic: trigger search for questions, current events, or specific factual queries
    query_lower = query.lower()
    keywords = [
        "what is", "who is", "when is", "where is", "how to",
        "latest news", "current events", "update on", "recent developments",
        "today", "this week", "this month", "in 20", # Catches years like "in 2025"
        "information about", "details on", "explain", "tell me about",
        "events in", "report on", "facts about", "background on"
    ]
    
    for keyword in keywords:
        if keyword in query_lower:
            return True
    
    # If the query is short and seems like a factual question, consider searching
    if len(query.split()) < 7 and query.endswith("?"):
        return True
        
    return False

def store_message(message, embedding):
    """Stores the message and its embedding in Qdrant."""
    point = PointStruct(
        id=int(time.time_ns()),  # Generate an integer ID based on nanosecond timestamp
        vector=embedding,
        payload={"text": message},
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[point])

def retrieve_relevant_messages(query, top_k=5):
    """Retrieves relevant messages from Qdrant based on the query."""
    query_embedding = generate_embedding(query)
    print("Query Embedding (first 10 elements):", query_embedding[:10])  # Print first 10 elements
    print("Query Embedding Length:", len(query_embedding))  # Print length of the embedding
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
    )
    messages = [hit.payload["text"] for hit in search_result]
    return messages

def generate_response(prompt, system_prompt=""):
    """Generates a response from the Gemini model, incorporating memory, web search results, and an optional system prompt."""
    
    web_search_results = ""
    web_search_performed = False
    if should_perform_web_search(prompt):
        web_search_performed = True
        search_placeholder = st.empty()
        with st.spinner("Searching the internet..."):
            web_search_results = perform_web_search(prompt)
        search_placeholder.empty() # Clear the spinner after search
        
    relevant_messages = retrieve_relevant_messages(prompt)
    memory_context = "\n".join(relevant_messages)
    
    # Combine contexts
    full_context = ""
    if web_search_performed:
        if web_search_results and web_search_results != "No relevant search results found.":
            full_context += f"Web Search Results:\n{web_search_results}\n\n"
        else:
            full_context += "Web Search Performed: No relevant results found.\n\n"
    if memory_context:
        full_context += f"Memory Context:\n{memory_context}\n\n"
        
    if system_prompt:
        augmented_prompt = f"System: {system_prompt}\n{full_context}User: {prompt}"
    else:
        augmented_prompt = f"{full_context}User: {prompt}"
        
    try:
        response = model.generate_content(augmented_prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit app
st.title("Gemini Chatbot with Memory")

# System Prompt input
st.subheader("System Prompt")
current_system_prompt = load_system_prompt()
system_prompt_input = st.text_area("Enter your system prompt here (optional):", value=current_system_prompt, height=100)

if st.button("Save System Prompt"):
    save_system_prompt(system_prompt_input)
    st.success("System prompt saved!")
    current_system_prompt = system_prompt_input # Update current_system_prompt after saving

st.subheader("Chat")
user_input = st.text_input("Enter your message:")

if st.button("Send"):
    if user_input:
        embedding = generate_embedding(user_input)
        store_message(user_input, embedding)
        response = generate_response(user_input, system_prompt=current_system_prompt)
        st.write("Response:", response)
    else:
        st.write("Please enter a message.")

st.subheader("Gmail Reader")

if 'show_gmail_reader' not in st.session_state:
    st.session_state.show_gmail_reader = False

if st.button("Enable Gmail Reading"):
    st.session_state.show_gmail_reader = not st.session_state.show_gmail_reader

if st.session_state.show_gmail_reader:
    if not GOOGLE_EMAIL_ADDRESS or not GOOGLE_APP_PASSWORD:
        st.warning("Gmail email address or app password not set in .env. Cannot initialize Gmail Reader.")
    else:
        gmail_reader = GmailReader(GOOGLE_EMAIL_ADDRESS, GOOGLE_APP_PASSWORD)
        if not gmail_reader.mail:
            st.error("Gmail Reader could not be initialized. Please check your email address and app password.")
        else:
            email_processor = EmailQueryProcessor(gmail_reader)
            st.write("Enter your email query:")
            user_email_query = st.text_input("Email Query:")
            if st.button("Process Email Query"):
                if user_email_query:
                    with st.spinner("Processing your email query..."):
                        processed_results = email_processor.process_email_query(user_email_query, model)
                    st.subheader("Email Query Results")
                    st.write(processed_results)
                else:
                    st.write("Please enter an email query.")
