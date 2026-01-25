import os
import random
import time

import certifi
from chromadb.config import Settings
from dotenv import load_dotenv
from google.cloud import aiplatform_v1
from google.cloud.aiplatform_v1.types import PredictResponse
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer

from logger import log_info, log_warning

load_dotenv()

# Configure SSL context to use certifi certificates
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Use SentencePiece tokenizer of T5，closer to Vertex AI Embeddings tokenization logic
tokenizer = AutoTokenizer.from_pretrained("t5-base")

def count_tokens(text: str) -> int:
    """Count the number of tokens"""
    encodings = tokenizer(text, truncation=False, add_special_tokens=False)
    return len(encodings["input_ids"])

class VertexAIEmbeddings(Embeddings):
    """Custom embedding class using Vertex AI PredictionServiceClient.

    Provides methods to embed documents and queries with retry logic,
    batch handling, and token limit enforcement.
    """
    def __init__(
        self,
        project: str = "genai-llmops-repo1",
        location: str = "us-central1",
        model: str = "text-embedding-005",
        batch_size: int = 15,
        max_tokens: int = 17500,
        retry_min_seconds: int = 10,
        max_retries: int = 3,
        show_progress_bar: bool = False,
    ) -> None:
        """Initialize VertexAIEmbeddings with configuration parameters."""
        self.endpoint = (
            f"projects/{project}/locations/{location}/publishers/google/models/{model}"
        )
        self.batch_size = batch_size
        self.retry_min_seconds = retry_min_seconds
        self.max_retries = max_retries
        self.show_progress_bar = show_progress_bar
        self.max_tokens = max_tokens

    def _predict_with_retry(self, instances: list[dict]) -> PredictResponse:
        """Call Vertex AI prediction API with retry logic.

        Args:
            instances: List of input instances for prediction.

        Returns:
            PredictResponse: Prediction response object from Vertex AI.

        Raises:
            Exception: If all retries fail.
            RuntimeError: Added as a safeguard; should not normally be reached.
        """
        for attempt in range(self.max_retries + 1):
            try:
                client = aiplatform_v1.PredictionServiceClient()
                return client.predict(endpoint=self.endpoint, instances=instances)
            except Exception as e:
                if attempt==self.max_retries:
                    log_warning(
                        f"Error {type(e).__name__} in Embeddings: {str(e)[:200]}. "
                        f"Max retries exceeded"
                    )
                    raise
                wait_time = (
                    self.retry_min_seconds * (2 ** attempt) + random.uniform(0, 1)
                )
                log_warning(
                    f"Error {type(e).__name__} in Embeddings: {str(e)[:200]}. "
                    f"Retrying in {wait_time:.1f}s "
                    f"(attempt {attempt+1}/{self.max_retries})"
                )
                time.sleep(wait_time)
                continue
        
        raise RuntimeError("Prediction failed without returning a response.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents into vector representations.
        
        Args:
            texts: List of input documents as strings.
        
        Returns:
            List of embeddings, each embedding is a list of floats.
        """
        results = []
        i = 0
        while i < len(texts):
            current_batch_size = self.batch_size
            while current_batch_size > 0:
                batch = texts[i:i+current_batch_size]
                instances = [
                    {"task_type": "RETRIEVAL_DOCUMENT", "content": t} for t in batch
                ]

                # Count the total number of tokens
                total_tokens = sum(count_tokens(t) for t in batch)

                if total_tokens <= self.max_tokens:
                    response = self._predict_with_retry(instances)
                    batch_embeddings = [
                        list(pred["embeddings"]["values"]) 
                        for pred in response.predictions
                    ]
                    results.extend(batch_embeddings)
                    i += current_batch_size
                    break
                else:
                    # Reduce batch size if exceeding max_tokens
                    current_batch_size -= 1

            if current_batch_size == 0:
                log_warning(
                    f"Skipping document at index {i}, "
                    f"exceeds {self.max_tokens} tokens alone"
                )
                i += 1
        return results

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query into a vector representation.
        
        Args:
            text: Query string.
        
        Returns:
            Embedding vector as a list of floats.
        
        Raises:
            ValueError: If query exceeds max token limit.
        """
        total_tokens = count_tokens(text)
        if total_tokens > self.max_tokens:
            raise ValueError(
                f"Query too long: {total_tokens} tokens (limit {self.max_tokens})"
            )
        instances = [{"task_type": "RETRIEVAL_QUERY", "content": text}]
        response = self._predict_with_retry(instances)
        return list(response.predictions[0]["embeddings"]["values"])

embeddings = VertexAIEmbeddings(
    project="genai-llmops-repo1",
    location="us-central1",
    model = "text-embedding-005", 
    batch_size=15, 
    max_tokens=17500,  # set with a safe margin to Vertex AI API upper limit 20,000
    retry_min_seconds=10, 
    max_retries=3, 
    show_progress_bar=False,
)

provider = os.getenv("VECTORSTORE_PROVIDER", "chroma").lower()

if provider == "chroma":
    client_settings = Settings(
        anonymized_telemetry=False,
        persist_directory=os.environ["CHROMA_PERSIST_DIR"],
    )

    vectorstore = Chroma(
        embedding_function=embeddings,
        client_settings=client_settings,
        collection_name="docs",
        persist_directory=os.environ["CHROMA_PERSIST_DIR"],
    )
elif provider == "pinecone":
    # Initialize Pinecone client
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX_NAME"]

    # Check if index already exists
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    # Create a serverless index if index does not exist
    if index_name not in existing_indexes:
        log_info(f"✨ Index '{index_name}' does not exist，create new index")
        pc.create_index(
            name=index_name,
            dimension=768,  # MUST be the same as your embedding dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",          # Options: "aws", "gcp"
                region="us-east-1"    # Region (us-central1 for gcp)
            )
        )

    # Load vector store from index
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
else:
    raise ValueError(f"Unsupported VECTORSTORE_PROVIDER: {provider}")
