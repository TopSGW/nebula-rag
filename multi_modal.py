import sys
import logging

from llama_index.multi_modal_llms.ollama import OllamaMultiModal
from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import SimpleMultiModalQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
import qdrant_client

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Initialize the multi-modal model with specific kwargs to handle responses
mm_model = OllamaMultiModal(
    model="llava:34b",
    request_timeout=120.0,
    temperature=0.1,
    system_prompt="You are a helpful assistant that answers questions about images and text."
)

# Initialize Qdrant client
client = qdrant_client.QdrantClient(path="qdrant_mm_db")

# Set up vector stores for text and images
text_store = QdrantVectorStore(
    client=client,
    collection_name="text_collection"
)

image_store = QdrantVectorStore(
    client=client,
    collection_name="image_collection"
)

# Initialize storage context
storage_context = StorageContext.from_defaults(
    vector_store=text_store,
    image_store=image_store
)

# Initialize CLIP embedding model for images
image_embed_model = ClipEmbedding()

# Initialize local text embedding model
text_embed_model = embed_model = OllamaEmbedding(
    model_name="llama3.3:70b",
    base_url="http://localhost:11434",
)


try:
    # Load documents
    documents = SimpleDirectoryReader("./mixed_wiki/").load_data()

    # Create multi-modal index with both embedding models specified
    mm_index = MultiModalVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        image_embed_model=image_embed_model,
        embed_model=text_embed_model  # Explicitly set text embedding model
    )

    # Define QA template with simpler format
    qa_tmpl_str = (
        "Context: {context_str}\n"
        "Question: {query_str}\n"
        "Answer: "
    )

    qa_tmpl = PromptTemplate(qa_tmpl_str)

    # Create query engine with specific kwargs
    mm_query_engine = mm_index.as_query_engine(
        llm=mm_model,
        text_qa_template=qa_tmpl,
        streaming=False,  # Disable streaming to avoid response handling issues
        similarity_top_k=2  # Limit the number of retrieved documents
    )

    # Test query
    query_str = "Tell me more about the Porsche"
    try:
        response = mm_query_engine.query(query_str)
        print("Response:", str(response))
    except Exception as e:
        print(f"Query error: {str(e)}")

except Exception as e:
    print(f"Setup error: {str(e)}")