# ref: https://github.com/microsoft/autogen/blob/main/autogen/agentchat/contrib/qdrant_retrieve_user_proxy_agent.py
# ref: https://github.com/microsoft/autogen/blob/main/autogen/retrieve_utils.py 

from typing import Callable, Dict, List, Optional, Union
import logging
import os

import glob
from urllib.parse import urlparse
import requests
import pypdf

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.fastembed_common import QueryResponse
    import fastembed
except ImportError as e:
    logging.fatal("Failed to import qdrant_client with fastembed. Try running 'pip install qdrant_client[fastembed]'")
    raise e

# ref: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb 
def count_token(input_string: str):
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base") # gpt-4, gpt-3.5-turbo, text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
    token_integers = encoding.encode(input_string)
    num_tokens = len(token_integers)
    return num_tokens


def split_text_to_chunks(
    text: str,
    max_tokens: int = 4000,
    overlap: int = 10,
):
    """Split a long text into chunks of max_tokens."""
    chunks = []
    lines = text.split("\n")
    lines_tokens = [count_token(line) for line in lines]
    sum_tokens = sum(lines_tokens)
    while sum_tokens > max_tokens:
        estimated_line_cut = int(max_tokens / sum_tokens * len(lines)) + 1
        cnt = 0
        prev = ""
        for cnt in reversed(range(estimated_line_cut)):
            if lines[cnt].strip() != "":
                continue
            if sum(lines_tokens[:cnt]) <= max_tokens:
                prev = "\n".join(lines[:cnt])
                break
        if cnt == 0:
            logger.warning(
                f"max_tokens is too small to fit a single line of text. Breaking this line:\n\t{lines[0][:100]} ..."
            )

        split_len = int(max_tokens / lines_tokens[0] * 0.9 * len(lines[0]))
        prev = lines[0][:split_len]
        lines[0] = lines[0][split_len:]
        lines_tokens[0] = count_token(lines[0])

        chunks.append(prev) if len(prev) > 10 else None  # don't add chunks less than 10 characters
        lines = lines[cnt:]
        lines_tokens = lines_tokens[cnt:]
        sum_tokens = sum(lines_tokens)
    text_to_chunk = "\n".join(lines)
    chunks.append(text_to_chunk) if len(text_to_chunk) > 10 else None  # don't add chunks less than 10 characters
    return chunks

def extract_text_from_pdf(file: str) -> str:
    """Extract text from PDF files"""
    text = ""
    with open(file, "rb") as f:
        reader = pypdf.PdfReader(f)
        if reader.is_encrypted:  # Check if the PDF is encrypted
            try:
                reader.decrypt("")
            except pypdf.errors.FileNotDecryptedError as e:
                logger.warning(f"Could not decrypt PDF {file}, {e}")
                return text  # Return empty text if PDF could not be decrypted

        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()

    if not text.strip():  # Debugging line to check if text is empty
        logger.warning(f"Could not decrypt PDF {file}")

    return text

def split_files_to_chunks(
    files: list,
    max_tokens: int = 4000
):
    """Split a list of files into chunks of max_tokens."""

    chunks = []

    for file in files:
        _, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()

        if file_extension == ".pdf":
            text = extract_text_from_pdf(file)
        else:  # For non-PDF text-based files
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        if not text.strip():  # Debugging line to check if text is empty after reading
            logger.warning(f"No text available in file: {file}")
            continue  # Skip to the next file if no text is available

        chunks += split_text_to_chunks(text, max_tokens)

    return chunks

def is_url(string: str):
    """Return True if the string is a valid URL."""
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def get_files_from_dir(dir_path: Union[str, List[str]], types: list, recursive: bool = True):
    """Return a list of all the files in a given directory, a url, a file path or a list of them."""
    if len(types) == 0:
        raise ValueError("types cannot be empty.")
    types = [t[1:].lower() if t.startswith(".") else t.lower() for t in set(types)]
    types += [t.upper() for t in types]

    files = []
    # If the path is a list of files or urls, process and return them
    if isinstance(dir_path, list):
        for item in dir_path:
            if os.path.isfile(item):
                files.append(item)
            elif is_url(item):
                files.append(get_file_from_url(item))
            elif os.path.exists(item):
                try:
                    files.extend(get_files_from_dir(item, types, recursive))
                except ValueError:
                    logger.warning(f"Directory {item} does not exist. Skipping.")
            else:
                logger.warning(f"File {item} does not exist. Skipping.")
        return files

    # If the path is a file, return it
    if os.path.isfile(dir_path):
        return [dir_path]

    # If the path is a url, download it and return the downloaded file
    if is_url(dir_path):
        return [get_file_from_url(dir_path)]

    if os.path.exists(dir_path):
        logger.info(f"Processing {dir_path}")
        for type in types:
            if recursive:
                files += glob.glob(os.path.join(dir_path, f"**/*.{type}"), recursive=True)
            else:
                files += glob.glob(os.path.join(dir_path, f"*.{type}"), recursive=False)
    else:
        logger.error(f"Directory {dir_path} does not exist.")
        raise ValueError(f"Directory {dir_path} does not exist.")
    print(dir_path)
    return files

def get_file_from_url(url: str, save_path: str = None):
    """Download a file from a URL."""
    if save_path is None:
        os.makedirs("/tmp/rag_files_from_url", exist_ok=True)
        save_path = os.path.join("/tmp/rag_files_from_url", os.path.basename(url))
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return save_path

def create_qdrant_collection_from_dir(
    dir_path: Union[str, List[str]], 
    text_types: list,
    max_tokens: int = 4000,
    client: QdrantClient = None,
    collection_name: str = "default_collection",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    payload_indexing: bool = False,
):
    if client is None:
        client = QdrantClient()
        client.set_model(embedding_model)
    
    texts = get_files_from_dir(dir_path, text_types, recursive=True)
    chunks = split_files_to_chunks(texts, max_tokens)
    logger.info(f"Found {len(chunks)} chunks.")

    collection = None
    # Check if collection by same name exists, if not, create it with custom options
    try:
        collection = client.get_collection(collection_name=collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=client.get_fastembed_vector_params(), # using default values
        )
        collection = client.get_collection(collection_name=collection_name)
    
    # Upsert in batch of 100 or less if the total number of chunks is less than 100
    for i in range(0, len(chunks), min(100, len(chunks))):
        end_idx = i + min(100, len(chunks) - i)
        client.add(
            collection_name,
            documents=chunks[i:end_idx],
            ids=[j for j in range(i, end_idx)] # removed length from extra_docs option
        )

    # Create a payload index for the document field
    # Enables highly efficient payload filtering. Reference: https://qdrant.tech/documentation/concepts/indexing/#indexing
    if payload_indexing:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="document",
            field_schema=models.TextIndexParams(
                type="text",
                tokenizer=models.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=15,
            ),
        )

def query_qdrant(
    query_texts: List[str],
    n_results: int = 10,
    client: QdrantClient = None,
    collection_name: str = "default_collection",
    search_string: str = "",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
) -> List[List[QueryResponse]]:
    
    if client is None:
        client = QdrantClient()
        client.set_model(embedding_model)

    results = client.query_batch(
        collection_name,
        query_texts,
        limit=n_results,
        query_filter=(
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="document",
                        match=models.MatchText(text=search_string),
                    )
                ]
            )
            if search_string
            else None
        ),
    )

    data = {
        "ids": [[result.id for result in sublist] for sublist in results],
        "documents": [[result.document for result in sublist] for sublist in results],
    }
    return data

def retrieve_docs(
    query_texts: str = "",
    client: QdrantClient = QdrantClient(":memory:"),
    collection_name: str = "default_collection",
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    payload_indexing: bool = False,
    n_results: int = 20,
    search_string: str = ""
    ):

    results = query_qdrant(
        query_texts=query_texts,
        n_results=n_results,
        search_string=search_string,
        client=client,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    return results

def main():
    # create_qdrant_collection_from_dir("https://raw.githubusercontent.com/microsoft/flaml/main/README.md",
    #         "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Research.md")
    
    retrieve_docs(query_texts = "Is there a function called tune_automl?", client=QdrantClient())

if __name__ == "__main__":
    main()