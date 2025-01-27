import os
import logging
import sys
from dotenv import load_dotenv

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    KnowledgeGraphIndex, 
    load_index_from_storage,
    Settings,
)

from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core.storage import StorageContext
from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Load environment variables from .env file
load_dotenv()

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

graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags
)

storage_context = StorageContext.from_defaults(graph_store=graph_store)

documents = SimpleDirectoryReader("./data/blackrock").load_data()

kg_index = KnowledgeGraphIndex.from_documents(
    documents=documents,
    storage_context=storage_context,
    max_triplets_per_chunk=10,
    rel_prop_names=rel_prop_names,
    tags=tags
)

nl2kg_query_engine = KnowledgeGraphQueryEngine(
    storage_context=storage_context,
    llm=Settings.llm
)

question = """
Who are founders of BlackRock?
"""

response_nl2kg = nl2kg_query_engine.query(question)

print("The Cypher Query is:")

query_string = nl2kg_query_engine.generate_query(question)

print( 
        f"""
```cypher
{query_string}
```
"""
)

print("The Answer is:")
print(response_nl2kg)