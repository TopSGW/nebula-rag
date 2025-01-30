from llama_index.multi_modal_llms.ollama import OllamaMultiModal

mm_model = OllamaMultiModal(model="llava:34b")

from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.embeddings.clip import ClipEmbedding

from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import SimpleMultiModalQueryEngine
import qdrant_client
from llama_index.embeddings.ollama import OllamaEmbedding

text_embed_model = OllamaEmbedding(
    model_name="llama3.3:70b",
    base_url="http://localhost:11434",
)

client = qdrant_client.QdrantClient(path="qdrant_mm_db")

text_store = QdrantVectorStore(
    client=client,
    collection_name="text_collection"
)

image_store = QdrantVectorStore(
    client=client,
    collection_name="image_collection"
)

storage_context = StorageContext.from_defaults(
    vector_store=text_store,
    image_store=image_store
)

image_embed_model = ClipEmbedding()

documents = SimpleDirectoryReader("./mixed_wiki/").load_data()

mm_index = MultiModalVectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    image_embed_model=image_embed_model,    
    embed_model=text_embed_model
)

qa_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

qa_tmpl = PromptTemplate(qa_tmpl_str)

mm_query_engine = mm_index.as_query_engine(llm=mm_model, text_qa_template=qa_tmpl)

query_str = "Tell me more about the Porsche"

response = mm_query_engine.query(query_str)

print(str(response))