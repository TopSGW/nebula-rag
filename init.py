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
    PropertyGraphIndex,
    Settings,
)

from llama_index.core.storage import StorageContext
from llama_index.graph_stores.nebula import NebulaGraphStore, NebulaPropertyGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from lancedb.rerankers import ColbertReranker
# Load environment variables from .env file
load_dotenv()

os.environ['NEBULA_USER'] = os.getenv('NEBULA_USER')
os.environ['NEBULA_PASSWORD'] = os.getenv('NEBULA_PASSWORD')
os.environ['NEBULA_ADDRESS'] = os.getenv('NEBULA_ADDRESS')

# reranker = ColbertReranker()

# vector_store = LanceDBVectorStore(
#     uri="./lancedb", mode="overwrite", reranker=reranker
# )

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

graph_store = NebulaPropertyGraphStore(
    space="llamaindex_nebula_property_graph", overwrite=True
)

vector_store = SimpleVectorStore()

documents = SimpleDirectoryReader("./data/blackrock").load_data()

# Initialize PropertyGraphIndex
pg_index = PropertyGraphIndex.from_documents(
    documents=documents,
    property_graph_store=graph_store,
    vector_store=vector_store,
    show_progress=True
)
pg_index.storage_context.vector_store.persist("./storage_graph/nebula_vec_store.json")

question = "Who are the founders of BlackRock?"

query_engine = pg_index.as_query_engine(
    llm=Settings.llm,
    include_text=True
)

query_response = query_engine.query(question)
print(f"{question}")

print("The response of query is:")

print(query_response)
q2_response = query_engine.query("How did Larry Fink and Rob Kapito meet?")
print("How did Larry Fink and Rob Kapito meet?")  
print(q2_response)