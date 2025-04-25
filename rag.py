from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
import os
import json
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from config import NVIDIA_API_KEY, EMBEDDING_MODEL, LLM_MODEL, FILES_TO_INDEX

# Configure logging
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)
maximum_threads = 13

# Suppress warnings about missing deep learning frameworks
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# Create directory for storing the index if it doesn't exist
index_dir = "index_storage"
if not os.path.exists(index_dir):
    os.makedirs(index_dir)

# Set NVIDIA API key in environment
os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY

def get_file_hash(file_path):
    """Generate a unique hash for a file based on its path and content."""
    with open(file_path, 'rb') as f:
        file_content = f.read()
    return hashlib.md5(file_content).hexdigest()

def get_index_path(file_path):
    """Get the index path for a given file."""
    file_hash = get_file_hash(file_path)
    return os.path.join(index_dir, f"index_{file_hash}")

def get_index_mapping():
    """Get the current index mapping."""
    mapping_file = os.path.join(index_dir, "index_mapping.json")
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            return json.load(f)
    return {}

def update_index_mapping(file_path, index_path, generated_files):
    """Update the index mapping with new index information."""
    mapping_file = os.path.join(index_dir, "index_mapping.json")
    mapping = get_index_mapping()
    
    file_hash = get_file_hash(file_path)
    mapping[file_path] = {
        "hash": file_hash,
        "index_path": os.path.basename(index_path),
        "timestamp": os.path.getmtime(file_path),
        "generated_files": generated_files
    }
    
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    #logger.info(f"Updated index mapping for {os.path.basename(file_path)}")

def is_index_valid(file_path, index_path):
    """Check if the existing index is valid and up to date."""
    try:
        # Check if index directory exists
        if not os.path.exists(index_path):
            return False
        print(f"Index path exists: {index_path}")
        
        # Get the mapping information
        mapping = get_index_mapping()
        print(f"Mapping: {mapping}")
        if file_path not in mapping:
            return False
        
        mapping_info = mapping[file_path]
        
        # Check if all required files exist
        required_files = mapping_info.get("generated_files", [])
        print(f"Required files: {required_files}")
        for file in required_files:
            if not os.path.exists(os.path.join(index_path, file)):
                return False
            print(f"File exists: {os.path.join(index_path, file)}")
        
        # Check if the file has been modified since index creation
        index_mtime = mapping_info["timestamp"]
        file_mtime = os.path.getmtime(file_path)
        
        if file_mtime > index_mtime:
            return False
        
        # Check if the file hash matches
        current_hash = get_file_hash(file_path)
        if current_hash != mapping_info["hash"]:
            return False
        
        return True
    except Exception as e:
        #logger.error(f"Error checking index validity: {str(e)}")
        return False

def load_index(file_path, index_path):
    """Load an existing index from storage."""
    try:
        print(f"Loading existing index from {index_path}")
        
        # Create storage context with persist directory
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        
        # Load index using the storage context
        try:
            index = load_index_from_storage(storage_context)
            print(f"Successfully loaded existing index from {index_path}")
            return index
        except Exception as vector_error:
            print(f"Error loading using vector store: {str(vector_error)}")
            
        
        # try:
        #     index = VectorStoreIndex.load_from_disk(
        #         index_path,
        #         storage_context=storage_context
        #     )
        #     print(f"Successfully loaded existing index from {index_path}")
        #     return index
        # except Exception as load_error:
        #     print(f"Error loading index from disk: {str(load_error)}")
        #     print("Attempting to load using vector store directly...")
        #     try:
        #         # Try loading using vector store directly
        #         # index = VectorStoreIndex.from_vector_store(
        #         #     storage_context.vector_store,
        #         #     storage_context=storage_context
        #         # )
        #         # print("Successfully loaded index using vector store directly")
        #         # return index
        #         index = load_index_from_storage(storage_context)
        #         print(f"Successfully loaded existing index from {index_path}")
        #         return index
        #     except Exception as vector_error:
        #         print(f"Error loading using vector store: {str(vector_error)}")
        #         return None
                
    except Exception as e:
        print(f"Failed to load existing index: {str(e)}")
        return None

def create_index(file_path, index_path):
    """Create a new index for the given file."""
    try:
        print(f"Creating new index for {file_path}")
        
        # Load documents
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        print(f"Loaded {len(documents)} documents")
        
        # Create storage context
        storage_context = StorageContext.from_defaults()
        
        # Create index
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        # Persist the index
        print(f"Persisting index to {index_path}")
        index.storage_context.persist(persist_dir=index_path)
        
        # Verify the index was persisted
        if not os.path.exists(os.path.join(index_path, 'docstore.json')):
            raise Exception("Failed to persist docstore")
        if not os.path.exists(os.path.join(index_path, 'vector_store.json')):
            raise Exception("Failed to persist vector store")
            
        print(f"Successfully created and persisted index to {index_path}")
        return index
    except Exception as e:
        print(f"Failed to create index: {str(e)}")
        return None

def load_or_create_index(file_path):
    """Load existing index or create a new one for a file."""
    try:
        index_path = get_index_path(file_path)
        index = None
        
        # Check if index directory exists
        if os.path.exists(index_path):
            print(f"Index path exists: {index_path}")
            # Try to load existing index
            index = load_index(file_path, index_path)
            if index is not None:
                return index
            print("Failed to load existing index, creating new one...")
        
        # Create new index if loading failed or index doesn't exist
        index = create_index(file_path, index_path)
        if index is None:
            raise Exception("Failed to create new index")
            
        return index
    except Exception as e:
        print(f"Error in load_or_create_index for {file_path}: {str(e)}")
        return None

def process_file(file_path: str) -> Optional[VectorStoreIndex]:
    """Process a single file to load or create its index."""
    try:
        return load_or_create_index(file_path)
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

class CombinedRetriever:
    def __init__(self, retrievers: Dict[str, VectorIndexRetriever]):
        self.retrievers = retrievers
        self.executor = ThreadPoolExecutor(max_workers=min(len(retrievers), maximum_threads))
    
    def retrieve_from_single(self, file_path: str, retriever: VectorIndexRetriever, query: str) -> Tuple[str, List]:
        """Retrieve results from a single retriever."""
        try:
            print(f"Retrieving from {os.path.basename(file_path)}")
            results = retriever.retrieve(query)
            print(f"Found {len(results)} relevant chunks from {os.path.basename(file_path)}")
            return file_path, results
        except Exception as e:
            print(f"Error retrieving from {os.path.basename(file_path)}: {str(e)}")
            return file_path, []
    
    def retrieve(self, query: str) -> List:
        """Retrieve results from all retrievers in parallel."""
        try:
            # Submit all retrieval tasks
            future_to_retriever = {
                self.executor.submit(
                    self.retrieve_from_single, 
                    file_path, 
                    retriever, 
                    query
                ): file_path 
                for file_path, retriever in self.retrievers.items()
            }
            
            # Collect results as they complete
            all_results = []
            for future in as_completed(future_to_retriever):
                file_path = future_to_retriever[future]
                try:
                    _, results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    print(f"Error processing results from {os.path.basename(file_path)}: {str(e)}")
            
            # Sort all results by similarity score
            all_results.sort(key=lambda x: x.score, reverse=True)
            print(f"Total relevant chunks found: {len(all_results)}")
            return all_results
            
        except Exception as e:
            print(f"Error in parallel retrieval: {str(e)}")
            return []
    
    def __del__(self):
        """Clean up the executor when the object is destroyed."""
        self.executor.shutdown(wait=True)

def setup_rag_system():
    """Set up the complete RAG system with specified models and files."""
    try:
        print("Starting RAG system setup...")
        
        # Initialize embedding model
        Settings.embed_model = NVIDIAEmbedding(model=EMBEDDING_MODEL)
        print("Embedding model initialized successfully")
        
        # Initialize text splitter
        Settings.node_parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        print("Text splitter initialized successfully")
        
        # Initialize LLM
        nvidia_llm = NVIDIA(model=LLM_MODEL)
        print("LLM initialized successfully")
        
        # Process files in parallel
        indices: Dict[str, VectorStoreIndex] = {}
        with ThreadPoolExecutor(max_workers=min(len(FILES_TO_INDEX), maximum_threads)) as executor:
            # Submit all files for processing
            future_to_file = {
                executor.submit(process_file, file_path): file_path 
                for file_path in FILES_TO_INDEX 
                if os.path.exists(file_path)
            }
            
            # Process results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    index = future.result()
                    if index is not None:
                        indices[file_path] = index
                        print(f"Successfully processed {os.path.basename(file_path)}")
                    else:
                        print(f"Failed to process {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
        
        if not indices:
            raise ValueError("No valid indices were created. Please check the input files and try again.")
        
        # Create retrievers for each index
        retrievers = {}
        for file_path, index in indices.items():
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=3,
                verbose=True
            )
            retrievers[file_path] = retriever
            print(f"Created VectorIndexRetriever for {os.path.basename(file_path)}")
        
        # Create combined retriever with parallel processing
        combined_retriever = CombinedRetriever(retrievers)
        print("Combined retriever created successfully")
        
        # Create chat engine
        chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=combined_retriever,
            llm=nvidia_llm,
            streaming=True,
            system_prompt="You are a helpful AI assistant specialized in compiler design. Use the provided context to answer questions accurately. If you don't know the answer, say so."
        )
        print("Chat engine created successfully")
        
        return chat_engine
    except Exception as e:
        print(f"Error setting up RAG system: {str(e)}")
        raise

def chat_with_rag():
    print("Welcome to the Compiler Design Chatbot! Type 'quit' to exit.")
    print("You can ask questions about compiler design concepts from the provided document.")
    
    try:
        chat_engine = setup_rag_system()
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
                
            try:
                # Get streaming response
                streaming_response = chat_engine.stream_chat(user_input)
                
                # Print the response as it streams
                print("\nAssistant: ", end="")
                for token in streaming_response.response_gen:
                    print(token, end="", flush=True)
                print()
                
            except Exception as e:
                print(f"An error occurred: {str(e)}")
    except Exception as e:
        print(f"Failed to initialize chat engine: {str(e)}")

if __name__ == "__main__":
    chat_with_rag()
