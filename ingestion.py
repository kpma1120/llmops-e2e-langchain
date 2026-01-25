import asyncio
import random
import time
from collections import deque

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from logger import Colors, log_error, log_header, log_info, log_success, log_warning
from vectorstore import vectorstore

load_dotenv()

# Initialize Tavily components
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=3, max_breadth=8, max_pages=200)
tavily_crawl = TavilyCrawl()

# Limit to a maximum of 3 concurrent requests (< quota limit: 4)
semaphore = asyncio.Semaphore(3)

# Maximum of 20 requests per second (equivalent to 1200/min < quota limit: 1500/min)
MAX_REQUESTS_PER_SECOND = 20
last_request_times = deque(maxlen=MAX_REQUESTS_PER_SECOND)


async def index_documents_async(
    documents: list[Document], 
    batch_size: int = 15, 
    retry_min_seconds: int = 10,
    max_retries: int = 3,
) -> None:
    """Process documents in batches asynchronously with rate limiting, concurrency 
    control, and progress bar.

    Args:
        documents (list[Document]): List of LangChain Document objects to be indexed.
        batch_size (int, optional): Number of documents per batch. Defaults to 15.
        retry_min_seconds (int, optional): 
            Minimum wait time before retrying failed batch. Defaults to 10.
        max_retries (int, optional): 
            Maximum number of retries for failed batch. Defaults to 3.

    Returns:
        None: Results are logged; successful/failed batches are reported via logging.

    Raises:
        Exception: If vectorstore operations consistently fail after retries.
    """
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: "
        f"Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: "
        f"Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    async def add_batch(
        batch: list[Document], 
        batch_num: int, 
        retry_min_seconds: int = retry_min_seconds,
        max_retries: int = max_retries,
    ) -> bool:
        """Add a batch of documents to vectorstore with retry and exponential backoff.

        Args:
            batch (list[Document]): Documents to be added in this batch.
            batch_num (int): Batch index for logging.
            retry_min_seconds (int, optional): 
                Minimum wait time before retry. Defaults to outer scope value.
            max_retries (int, optional): 
                Maximum number of retries. Defaults to outer scope value.

        Returns:
            bool: True if batch added successfully, False otherwise.
        """
        for attempt in range(max_retries + 1):
            async with semaphore:
                # --- Rate limiting ---
                now = time.time()
                # Remove old records earlier than 1s
                while last_request_times and now - last_request_times[0] >= 1:
                    last_request_times.popleft()
                if len(last_request_times) >= MAX_REQUESTS_PER_SECOND:
                    sleep_time = 1 - (now - last_request_times[0])
                    log_info(
                        f"⏳ Rate limiter: "
                        f"Sleeping {sleep_time:.2f}s before batch {batch_num}",
                        Colors.YELLOW,
                    )
                    await asyncio.sleep(sleep_time)
                last_request_times.append(time.time())
                
                # --- Truly execute embedding ---
                try:
                    await vectorstore.aadd_documents(batch)
                    log_success(
                        f"VectorStore Indexing: "
                        f"Successfully added batch "
                        f"{batch_num}/{len(batches)} ({len(batch)} documents)"
                    )
                    return True
                except Exception as e:
                    if attempt==max_retries:
                        log_warning(
                            f"Error {type(e).__name__} "
                            f"on batch {batch_num}: {str(e)[:200]}. "
                            f"Max retries exceeded"
                        )
                        break
                    wait_time = (
                        retry_min_seconds * (2 ** attempt) + random.uniform(0, 1)
                    )
                    log_warning(
                        f"Error {type(e).__name__} "
                        f"on batch {batch_num}: {str(e)[:200]}. "
                        f"Retrying in {wait_time:.1f}s "
                        f"(attempt {attempt+1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue
        
        log_error(
            f"VectorStore Indexing: "
            f"Failed to add batch {batch_num} after {max_retries} retries"
        )
        return False
    
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = []
    for result in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Embedding batches",
        unit="batch",
    ):
        results.append(await result)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: "
            f"All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: "
            f"Processed {successful}/{len(batches)} batches successfully"
        )


async def main() -> None:
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )
    # Crawl the documentation site

    res = tavily_crawl.invoke(
        {
            "url": "https://docs.langchain.com/oss/python/langchain/overview",
            "max_depth": 2,
            "extract_depth": "basic",
        }
    )

    # Convert Tavily crawl results to LangChain Document objects
    all_docs = []
    for tavily_crawl_result_item in res["results"]:
        log_info(
            f"TavilyCrawl: "
            f"Successfully crawled {tavily_crawl_result_item['url']} "
            f"from documentation site"
        )
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: "
        f"Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: "
        f"Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(
        splitted_docs, batch_size=15, retry_min_seconds=10, max_retries=3
    )

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
