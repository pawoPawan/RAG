import os

# API Keys
NVIDIA_API_KEY = "api_key"

# Model Configuration
EMBEDDING_MODEL = "embedding_model"
LLM_MODEL = "inferencing_model"

# Directory Configuration
INDEX_DIR = "index_storage"
DOCUMENTS_DIR = "documents"

# Files to Index
FILES_TO_INDEX = [
    "/path/of/doc_1.pdf",
    "/path/of/doc_2.pdf",
    "/path/of/doc_3.pdf",
]

# Text Processing Configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Retriever Configuration
SIMILARITY_TOP_K = 3

# Chat Engine Configuration
SYSTEM_PROMPT = "You are a helpful AI assistant specialized in XYZ. Use the provided context to answer questions accurately. If you don't know the answer, say so." 
