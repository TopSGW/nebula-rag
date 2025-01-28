import os
import logging
import sys
from dotenv import load_dotenv

from llama_index.core import PropertyGraphIndex, StorageContext, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.property_graph.sub_retrievers.llm_synonym import LLMSynonymRetriever
from llama_index.core.indices.property_graph.sub_retrievers.vector import VectorContextRetriever
from llama_index.graph_stores.nebula import NebulaGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import nest_asyncio

nest_asyncio.apply()
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

graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags
)

storage_context = StorageContext.from_defaults(graph_store=graph_store)

# Initialize the PropertyGraphIndex
graph_index = PropertyGraphIndex.from_existing(storage_context=storage_context)

# Configure sub-retrievers
llm_synonym_retriever = LLMSynonymRetriever(
    graph_store=graph_store,
    llm=Settings.llm,
    include_text=True,
)

vector_context_retriever = VectorContextRetriever(
    graph_store=graph_store,
    embed_model=Settings.embed_model,
    include_text=True,
    similarity_top_k=2,
)

# Combine sub-retrievers
sub_retrievers = [llm_synonym_retriever, vector_context_retriever]

# Set up the query engine
retriever = graph_index.as_retriever(sub_retrievers=sub_retrievers)
query_engine = RetrieverQueryEngine.from_args(retriever)

# Execute a query
response = query_engine.query("Who are the founders of BlackRock?")

print('Answer is:')
print(response)
