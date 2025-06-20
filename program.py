import google.generativeai as genai
import os
import time
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np
from sentence_transformers import SentenceTransformer
from serpapi import GoogleSearch
from dotenv import load_dotenv
from gmail_reader import GmailReader
from email_query_processor import EmailQueryProcessor

load_dotenv()

SYSTEM_PROMPT_FILE = "system_prompt.txt"

def load_system_prompt():
    if os.path.exists(SYSTEM_PROMPT_FILE):
        with open(SYSTEM_PROMPT_FILE, "r") as f:
            return f.read()
    return ""

def save_system_prompt(prompt):
    with open(SYSTEM_PROMPT_FILE, "w") as f:
        f.write(prompt)

# API Keys from .env
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") # Load Qdrant API key from .env
GOOGLE_APP_PASSWORD = os.environ.get("GOOGLE_APP_PASSWORD")
GOOGLE_EMAIL_ADDRESS = os.environ.get("GOOGLE_EMAIL_ADDRESS")

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
except Exception: # Catch all exceptions for collection check
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' created.")

# Load Sentence Transformer model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
print(next(embedding_model.parameters()).device)

def generate_embedding(text):
    """Generate embedding for the given text using Sentence Transformers."""
    return embedding_model.encode(text).tolist()

def perform_web_search(query):
    """Performs a web search using SerpAPI and returns formatted results."""
    if not SERPAPI_API_KEY:
        print("SerpAPI key not set. Web search will not function.")
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
    query_lower = query.lower()
    keywords = [
        "what is", "who is", "when is", "where is", "how to",
        "latest news", "current events", "update on", "recent developments",
        "today", "this week", "this month", "in 20",
        "information about", "details on", "explain", "tell me about",
        "events in", "report on", "facts about", "background on"
    ]
    
    for keyword in keywords:
        if keyword in query_lower:
            return True
    
    if len(query.split()) < 7 and query.endswith("?"):
        return True
        
    return False

def store_message(message, embedding):
    """Stores the message and its embedding in Qdrant."""
    point = PointStruct(
        id=int(time.time_ns()),
        vector=embedding,
        payload={"text": message},
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[point])

def retrieve_relevant_messages(query, top_k=5):
    """Retrieves relevant messages from Qdrant based on the query."""
    query_embedding = generate_embedding(query)
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
        web_search_results = perform_web_search(prompt)
        
    relevant_messages = retrieve_relevant_messages(prompt)
    memory_context = "\n".join(relevant_messages)
    
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

def main():
    print("Gemini Chatbot with Memory (Command Line)")
    
    current_system_prompt = load_system_prompt()
    
    while True:
        print("\nMenu:")
        print("1. Chat with Gemini")
        print("2. View/Update System Prompt")
        print("3. Use Gmail Reader")
        print("4. Exit")
        
        choice = input("Enter your choice (1, 2, 3, or 4): ")
        
        if choice == '1':
            print("\nEntering chat mode.")
            while True:
                user_input = input("Enter your message (type 'exit' to escape): ")
                if user_input.lower() == 'exit':
                    break
                
                if user_input:
                    embedding = generate_embedding(user_input)
                    store_message(user_input, embedding)
                    response = generate_response(user_input, system_prompt=current_system_prompt)
                    print("Gemini:", response)
                else:
                    print("Please enter a message.")
        elif choice == '2':
            print(f"\nCurrent System Prompt:\n{current_system_prompt}")
            new_system_prompt = input("Enter new system prompt (leave blank to keep current): ")
            if new_system_prompt:
                save_system_prompt(new_system_prompt)
                current_system_prompt = new_system_prompt
                print("System prompt updated!")
            else:
                print("System prompt not changed.")
        elif choice == '3':
            print("\n--- Gmail Reader ---")
            if not GOOGLE_EMAIL_ADDRESS or not GOOGLE_APP_PASSWORD:
                print("Gmail email address or app password not set in .env. Cannot initialize Gmail Reader.")
                continue
            gmail_reader = GmailReader(GOOGLE_EMAIL_ADDRESS, GOOGLE_APP_PASSWORD)
            if not gmail_reader.mail:
                print("Gmail Reader could not be initialized. Please check your email address and app password.")
                continue

            email_processor = EmailQueryProcessor(gmail_reader)
            print("\nEntering Gmail Query mode.")
            while True:
                user_email_query = input("Enter your email query (type 'exit' to escape): ")
                if user_email_query.lower() == 'exit':
                    break
                
                if user_email_query:
                    print("Processing your email query...")
                    processed_results = email_processor.process_email_query(user_email_query, model)
                    print("\n--- Email Query Results ---")
                    print(processed_results)
                else:
                    print("Please enter an email query.")
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
