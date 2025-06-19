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

# Replace 'YOUR_API_KEY' with your actual Gemini API key
GEMINI_API_KEY = 'AIzaSyDwYGElMrX9pzSyK9QaccNdEl-W6ul2GGk'
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize Qdrant client
QDRANT_HOST = "https://741d8e2a-6524-4f1e-a9e0-0f7021ef7d51.us-east4-0.gcp.cloud.qdrant.io"
QDRANT_PORT = None # Port is part of the host URL
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.TXgCR1WzruH1qmbC5_NUjR2dhleZywhGssHkAEQPanA"

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
embedding_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
print(next(embedding_model.parameters()).device)

def generate_embedding(text):
    """Generate embedding for the given text using Sentence Transformers."""
    return embedding_model.encode(text).tolist()

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

def generate_response(prompt):
    """Generates a response from the Gemini model, incorporating memory."""
    relevant_messages = retrieve_relevant_messages(prompt)
    context = "\n".join(relevant_messages)
    augmented_prompt = f"Context: {context}\nUser: {prompt}"
    try:
        response = model.generate_content(augmented_prompt)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit app
st.title("Gemini Chatbot with Memory")

user_input = st.text_input("Enter your message:")

if st.button("Send"):
    if user_input:
        embedding = generate_embedding(user_input)
        store_message(user_input, embedding)
        response = generate_response(user_input)
        st.write("Response:", response)
    else:
        st.write("Please enter a message.")
