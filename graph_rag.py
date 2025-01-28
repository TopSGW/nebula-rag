import os
import logging
import sys
from dotenv import load_dotenv

from llama_index.core import PropertyGraphIndex, StorageContext, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import nest_asyncio

nest_asyncio.apply()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Initialize LLM and embedding model

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

Settings.embed_model = OllamaEmbedding(
    model_name="llama3.3:70b",
    base_url="http://localhost:11434",
)

# Configure graph store
space_name = "rag_workshop"
edge_types, rel_prop_names = ["relationship"], ["relationship"]
tags = ["entity"]

property_graph_store = NebulaPropertyGraphStore(
    space=space_name,
)

storage_context = StorageContext.from_defaults(property_graph_store=property_graph_store)

# Initialize the PropertyGraphIndex
graph_index = PropertyGraphIndex.from_existing(
    storage_context=storage_context,
    property_graph_store=property_graph_store
)


# Set up the query engine
retriever = graph_index.as_retriever(
    include_text=True,
    similarity_top_k=2,
)
query_engine = RetrieverQueryEngine.from_args(
    retriever,
    llm=Settings.llm
)


# Execute a query
response = query_engine.query("Who are the founders of BlackRock?")

print('Answer is:')
print(response)

query_engine2 = graph_index.as_query_engine(
    llm=Settings.llm
)
response2 = query_engine2.query("Who are the founders of BlackRock?")
print("Answer2 is :")
print(response2)