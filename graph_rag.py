import os
import logging
import sys
from dotenv import load_dotenv

from llama_index.core.storage import StorageContext
from llama_index.core import PropertyGraphIndex, Settings
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.memory import ChatMemoryBuffer
import nest_asyncio

nest_asyncio.apply()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Initialize LLM and embedding model

# Load environment variables from .env file
load_dotenv()

os.environ['NEBULA_USER'] = os.getenv('NEBULA_USER')
os.environ['NEBULA_PASSWORD'] = os.getenv('NEBULA_PASSWORD')
os.environ['NEBULA_ADDRESS'] = os.getenv('NEBULA_ADDRESS')

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

vec_store = SimpleVectorStore.from_persist_path("./storage_graph/nebula_vec_store.json")

property_graph_store = NebulaPropertyGraphStore(
    space="llamaindex_nebula_property_graph",
)

# Initialize the PropertyGraphIndex
graph_index = PropertyGraphIndex.from_existing(
    property_graph_store=property_graph_store,
    vector_store=vec_store,
    llm=Settings.llm,
    show_progress=True
)

# Execute a query
query_engine = graph_index.as_query_engine(
    llm=Settings.llm,
    include_text=True
)
response = query_engine.query("Who are the founders of BlackRock?")

print('Answer is:')
print(str(response))

memory = ChatMemoryBuffer.from_defaults(
    token_limit=1500,
    llm=Settings.llm
)

chat_engine = graph_index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    verbose=True,
    llm=Settings.llm
)

response = chat_engine.chat("who is Larry?")
print("who is Larry?")
print(response)
response = chat_engine.chat("who is Robert?")
print("who is Robert?")
print(response)
response = chat_engine.chat("who is Susan?")
print("who is Susan?")
print(response)

response = chat_engine.chat("How did Larry Fink and Rob Kapito meet?")
print("How did Larry Fink and Rob Kapito meet?")
print(response)