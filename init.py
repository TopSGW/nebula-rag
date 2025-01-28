import os
import logging
import sys
from dotenv import load_dotenv
import nest_asyncio

nest_asyncio.apply()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    PropertyGraphIndex,  # Updated import
    load_index_from_storage,
    Settings,
)

from llama_index.core.storage import StorageContext
from llama_index.graph_stores.nebula import NebulaGraphStore, NebulaPropertyGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.vector_stores.simple import SimpleVectorStore


# Load environment variables from .env file
load_dotenv()

os.environ['NEBULA_USER'] = os.getenv('NEBULA_USER')
os.environ['NEBULA_PASSWORD'] = os.getenv('NEBULA_PASSWORD')
os.environ['NEBULA_ADDRESS'] = os.getenv('NEBULA_ADDRESS')

print(f"text .env : NEBULA_ADDRESS: {os.getenv('NEBULA_USER')} : {os.getenv('NEBULA_ADDRESS')}")

Settings.llm = Ollama(
    model="llama3.3:70b",
    temperature=0.3,
    request_timeout=120.0,
    base_url="http://localhost:11434"
)

# Initialize embedding model
Settings.embed_model = OllamaEmbedding(
    model_name="llama3.3:70b",
    base_url="http://localhost:11434",
)

Settings.chunk_size = 512

space_name = "rag_workshop"
edge_types, rel_prop_names = ["relationship"], ["relationship"]
tags = ["entity"]

graph_store = NebulaPropertyGraphStore(
    space="llamaindex_nebula_property_graph", overwrite=True
)

storage_context = StorageContext.from_defaults(graph_store=graph_store)

documents = SimpleDirectoryReader("./data/blackrock").load_data()

vec_store = SimpleVectorStore()

# Initialize PropertyGraphIndex
pg_index = PropertyGraphIndex.from_documents(
    documents=documents,
    storage_context=storage_context,
    property_graph_store=graph_store,
    vector_store=vec_store,
    max_triplets_per_chunk=10,
    rel_prop_names=rel_prop_names,
    tags=tags,
    show_progress=True
)
pg_index.storage_context.vector_store.persist("./storage_graph/nebula_vec_store.json")

# Set up the retriever
retriever = pg_index.as_retriever(
    include_text=False,
    similarity_top_k=2,
)

question = "Who are the founders of BlackRock?"

print(f"{question}")

response = retriever.retrieve(question)

print("The Answer is:")
print(response)

query_engine = pg_index.as_query_engine(
    llm=Settings.llm,
    include_text=True
)

query_response = query_engine.query(question)
print(f"{question}")

print("The response of query is:")

print(query_response)