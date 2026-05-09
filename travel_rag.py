"""
CSC 7644 - LLM Application Development
Hybrid RAG Travel Assistant

Required Libraries:
    pip install chromadb rank-bm25 openai python-dotenv

Environment Variables (.env file):
    OPENAI_API_KEY=your_openai_api_key

Author: Sebastian Escobar-Mesa
Date: 5-9-2026
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import chromadb
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# PROVIDER CLIENT CONFIGURATION

def get_openai_client() -> OpenAI:
    """
    Create and return an OpenAI client using API key from environment.

    Returns:
        Configured OpenAI client instance.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=api_key)


# TEXT CHUNKING

def chunk_text(text: str, chunk_size: int, stride: int) -> List[str]:
    """
    Split text into overlapping chunks using character-level windowing.

    This is a simple but effective chunking strategy that ensures no information
    is lost at chunk boundaries by using overlapping windows.

    Args:
        text: The input text to chunk.
        chunk_size: Maximum number of characters per chunk.
        stride: Number of characters to advance between chunks.
                A stride < chunk_size creates overlap.

    Returns:
        List of text chunks.

    Example:
        >>> chunks = chunk_text("Hello world, this is a test.", 10, 5)
        >>> # Creates overlapping windows of 10 chars, advancing by 5
    """
    # Implement character-level chunking
    #
    # Initialize empty list for chunks
    chunks = []

    # Handle edge cases (empty text, invalid chunk_size or stride)
    if not text or chunk_size <= 0 or stride <= 0:
        return chunks

    # Use a while loop to slide a window across the text:
    start = 0  # Start at position 0
    while start < len(text):  # Continue while start < len(text)
        chunk = text[start:start + chunk_size]  # Extract substring from start to start + chunk_size
        if chunk.strip():
            chunks.append(chunk)  # Add chunk to list if non-empty (after stripping whitespace)
        start += stride  # Advance start by stride

    # Return the list of chunks
    return chunks


def infer_city(filename: str) -> str:
    """
    Infer the city name from the source filename.

    Args:
        filename: The name of the source file

    Returns:
        The city name or "Unknown" if no match found
    """
    name = filename.lower()

    if "houston" in name:
        return "Houston"
    if "atlanta" in name:
        return "Atlanta"
    if "los_angeles" in name or "la_" in name:
        return "Los Angeles"
    if "new_orleans" in name:
        return "New Orleans"
    if "new_york" in name or "nyc" in name:
        return "New York City"

    return "Unknown"


def detect_city_from_query(query: str) -> Optional[str]:
    """
    Detect whether a query references a supported city.

    Args:
        query: User query string

    Returns:
        Supported city name if found, otherwise None
    """
    q = query.lower()

    if "houston" in q:
        return "Houston"
    if "atlanta" in q:
        return "Atlanta"
    if "los angeles" in q or " la " in f" {q} ":
        return "Los Angeles"
    if "new orleans" in q:
        return "New Orleans"
    if "new york" in q or "nyc" in q:
        return "New York City"

    return None


def load_documents(data_dir: str) -> List[Dict[str, str]]:
    """
    Load all .txt files from a directory.

    Args:
        data_dir: Path to directory containing text files.

    Returns:
        List of dicts with 'filename' and 'content' keys.
    """
    documents = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for txt_file in data_path.glob("*.txt"):
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            documents.append({
                'filename': txt_file.name,
                'content': content
            })

    if not documents:
        print(f"Warning: No .txt files found in {data_dir}")

    return documents


# EMBEDDING FUNCTIONS

def get_embeddings(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAI's embedding API.

    Args:
        client: OpenAI client instance.
        texts: List of text strings to embed.
        model: Embedding model name (e.g., 'text-embedding-3-small').

    Returns:
        List of embedding vectors (each a list of floats).
    """
    if not texts:
        return []

    # OpenAI embedding API call
    response = client.embeddings.create(
        model=model,
        input=texts
    )

    # Extract embeddings in order
    embeddings = [item.embedding for item in response.data]

    return embeddings


# CHROMADB OPERATIONS

def get_chroma_client(db_path: str) -> chromadb.PersistentClient:
    """
    Create a persistent ChromaDB client.

    Args:
        db_path: Path to the database directory.

    Returns:
        ChromaDB PersistentClient instance.
    """
    return chromadb.PersistentClient(path=db_path)


def get_or_create_collection(
        chroma_client: chromadb.PersistentClient,
        collection_name: str
) -> chromadb.Collection:
    """
    Get an existing collection or create a new one.

    Args:
        chroma_client: ChromaDB client instance.
        collection_name: Name of the collection.

    Returns:
        ChromaDB Collection instance.
    """
    return chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )


def upsert_chunks(
        collection: chromadb.Collection,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: List[str]
) -> None:
    """
    Upsert chunks with their embeddings into ChromaDB.

    Upsert is idempotent: if an ID already exists, it will be updated.

    Args:
        collection: ChromaDB collection.
        chunks: List of text chunks.
        embeddings: List of embedding vectors.
        metadatas: List of metadata dicts for each chunk.
        ids: List of unique IDs for each chunk.
    """
    # The lengths should all be equal for the upsert to work correctly
    if not (len(chunks) == len(embeddings) == len(metadatas) == len(ids)):
        raise ValueError(
            f"Length mismatch: chunks={len(chunks)}, embeddings={len(embeddings)}, "
            f"metadatas={len(metadatas)}, ids={len(ids)}"
        )

    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )


# BM25 SEARCH

def tokenize_for_bm25(text: str) -> List[str]:
    """
    Simple tokenization for BM25: lowercase and split on whitespace.

    Args:
        text: Input text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    # Convert text to lowercase and split on whitespace
    return text.lower().split()


def build_bm25_index(documents: List[str]) -> BM25Okapi:
    """
    Build a BM25 index from a list of documents.

    Args:
        documents: List of document strings.

    Returns:
        BM25Okapi index object.
    """
    tokenized_docs = [tokenize_for_bm25(doc) for doc in documents]
    return BM25Okapi(tokenized_docs)


def bm25_search(
        bm25_index: BM25Okapi,
        query: str,
        documents: List[str],
        top_k: int
) -> List[Tuple[int, float, str]]:
    """
    Search using BM25 and return top-k results.

    Args:
        bm25_index: Pre-built BM25 index.
        query: Search query string.
        documents: Original document list (for returning text).
        top_k: Number of results to return.

    Returns:
        List of tuples: (doc_index, bm25_score, document_text)
    """
    # Tokenize the query using tokenize_for_bm25()
    tokenized_query = tokenize_for_bm25(query)

    # Get scores for all documents using bm25_index.get_scores(tokenized_query)
    scores = bm25_index.get_scores(tokenized_query)

    # Create list of (index, score) tuples for all documents
    indexed_scores = [(i, scores[i]) for i in range(len(scores))]

    # Sort by score in descending order
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    # Take top_k results and build output list with (index, score, document_text)
    results = []
    for i, score in indexed_scores[:top_k]:
        results.append((i, score, documents[i]))

    # Return the results list
    return results


# VECTOR SEARCH

def vector_search(
        collection: chromadb.Collection,
        query_embedding: List[float],
        top_k: int,
        city_filter: Optional[str] = None
) -> List[Tuple[str, float, str, Dict]]:
    """
    Search ChromaDB collection using vector similarity.

    Note: ChromaDB returns distances, not similarities. For cosine distance,
    similarity = 1 - distance.

    Args:
        collection: ChromaDB collection to search.
        query_embedding: Query vector.
        top_k: Number of results to return.
        city_filter: Filter results by city

    Returns:
        List of tuples: (id, similarity_score, document_text, metadata)
    """
    # ChromaDB query needs to include "documents", "distances", and "metadatas"
    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "distances", "metadatas"]
    }

    if city_filter:
        query_args["where"] = {"city": city_filter}

    results = collection.query(**query_args)

    ids = results['ids'][0] if results['ids'] else []
    documents = results['documents'][0] if results['documents'] else []
    distances = results['distances'][0] if results['distances'] else []
    metadatas = results['metadatas'][0] if results['metadatas'] else []

    output = []
    for i in range(len(ids)):
        similarity = 1.0 - distances[i]
        output.append((
            ids[i],
            similarity,
            documents[i],
            metadatas[i] if i < len(metadatas) else {}
        ))

    return output


# HYBRID FUSION

def normalize_scores(scores: List[float]) -> List[float]:
    """
    Min-max normalize scores to [0, 1] range.

    Args:
        scores: List of raw scores.

    Returns:
        List of normalized scores.
    """
    # Handle empty list case (return empty list)
    if not scores:
        return []

    # Find min and max values in scores
    min_score = min(scores)
    max_score = max(scores)

    # Handle case where all scores are the same (return list of 1.0s)
    if min_score == max_score:
        return [1.0] * len(scores)

    # Apply min-max normalization: (score - min) / (max - min)
    normalized = []
    for score in scores:
        normalized.append((score - min_score) / (max_score - min_score))

    # Return list of normalized scores
    return normalized


def hybrid_fusion(
        bm25_results: List[Tuple[int, float, str]],
        vector_results: List[Tuple[str, float, str, Dict]],
        alpha: float = 0.5
) -> List[Tuple[str, float, str]]:
    """
    Combine BM25 and vector search results using weighted score fusion.

    Args:
        bm25_results: Results from BM25 search (index, score, text).
        vector_results: Results from vector search (id, similarity, text, metadata).
        alpha: Weight for vector scores (1-alpha for BM25). Default 0.5.

    Returns:
        Fused results sorted by combined score: (id/index, fused_score, text)
    """
    # Normalize BM25 scores using normalize_scores()
    # Extract scores from bm25_results: [r[1] for r in bm25_results]
    bm25_scores = [r[1] for r in bm25_results]
    normalized_bm25 = normalize_scores(bm25_scores)

    # Normalize vector scores using normalize_scores()
    # Extract scores from vector_results: [r[1] for r in vector_results]
    vector_scores = [r[1] for r in vector_results]
    normalized_vector = normalize_scores(vector_scores)

    # Create a dictionary to track fused scores
    fused_dict = {}

    # Add BM25 results to the dictionary
    for (idx, score, text), normalized_score in zip(bm25_results, normalized_bm25):
        key = text[:150]
        fused_dict[key] = {
            "text": text,
            "bm25_score": normalized_score,
            "vector_score": 0.0,
            "id": str(idx)
        }

    # Add/update vector results in the dictionary
    for (doc_id, score, text, metadata), normalized_score in zip(vector_results, normalized_vector):
        key = text[:150]
        if key not in fused_dict:
            fused_dict[key] = {
                "text": text,
                "bm25_score": 0.0,
                "vector_score": normalized_score,
                "id": doc_id
            }
        else:
            fused_dict[key]["vector_score"] = normalized_score
            fused_dict[key]["id"] = doc_id

    # Compute weighted fusion: alpha * vector + (1 - alpha) * bm25
    results = []
    for item in fused_dict.values():
        fused_score = alpha * item["vector_score"] + (1 - alpha) * item["bm25_score"]

        # Build results list as (id, fused_score, text)
        results.append((item["id"], fused_score, item["text"]))

    # Sort by fused score descending and return
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ANSWER GENERATION
def remove_duplicate_chunks(chunks: List[str]) -> List[str]:
    """
    Remove repeated or near-identical chunks before LLM generation.

    Args:
        chunks: Retrieved text chunks

    Returns:
        List of unique chunks
    """
    unique_chunks = []
    seen = set()

    for chunk in chunks:
        key = chunk[:200].strip().lower()

        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)

    return unique_chunks


def format_context(chunks: List[str]) -> str:
    """
    Format retrieved chunks into a context string for the LLM.

    Args:
        chunks: List of retrieved text chunks.

    Returns:
        Formatted context string with numbered passages.
    """
    # Handle empty chunks (return "No relevant passages found.")
    if not chunks:
        return "No relevant passages found."

    # Format each chunk as "[Passage N]\n{chunk_text}"
    formatted_passage = []
    for i, chunk in enumerate(chunks, start=1):
        formatted_passage.append(f"[Passage {i}]\n{chunk}")

    # Join all formatted passages with "\n\n"
    context = "\n\n".join(formatted_passage)

    # Return the formatted context string
    return context


def generate_grounded_answer(
        client: OpenAI,
        query: str,
        context: str,
        model: str = "gpt-4o-mini"
) -> str:
    """
    Generate an answer grounded in the retrieved context.

    Args:
        client: OpenAI client instance.
        query: User's question.
        context: Retrieved passages formatted as context.
        model: Chat model to use.

    Returns:
        Generated answer string.
    """
    system_prompt = """
    You are a helpful travel planning assistant.

    You must answer using ONLY the provided travel passages.
    All recommendations must remain grounded in the retrieved context.

    Core Rules:
    - Do not use outside knowledge.
    - Do not invent attractions, restaurants, prices, hours,
      neighborhoods, transportation details, or events.
    - If the passages do not contain enough information, clearly say what
      information is missing.
    - If the user asks about a specific city, only use passages for that
      city.
    - Keep recommendations factual, grounded, and directly supported by the
      retrieved passages.
    - Prefer specific attractions, neighborhoods, restaurants, museums,
      outdoor activities, tours, landmarks, bars, markets, and local
      experiences over generic recommendations.
    - Avoid generic filler statements like:
      "there is something for everyone."

    Response Quality Rules:
    - Keep answers concise, informative, and useful for trip planning.
    - Avoid repeating the same attraction, landmark, neighborhood,
      restaurant, or recommendation multiple times in a response.
    - If a location has already been discussed, reference it briefly instead
      of fully repeating the description.
    - Combine overlapping information from multiple passages instead of
      repeating similar descriptions.
    - When multiple strong recommendations exist, vary which ones are
      emphasized first.
    - Do not always structure responses in exactly the same order.
    - Vary wording and sentence structure naturally while remaining accurate
      and grounded.
    - Avoid repetitive sentence openings and repetitive phrasing patterns.
    - Select a diverse mix of recommendations from the retrieved passages
      when appropriate.
    - Prefer variety across categories such as:
      outdoor activities, museums, beaches, nightlife, restaurants,
      shopping, entertainment, neighborhoods, parks, cultural attractions,
      and local experiences.
    - Only recommend items that directly match the user's request.
    - If the user asks about one category, avoid unrelated recommendations
      unless the passages clearly connect them naturally.
    - Avoid vague recommendations such as:
      "explore the area,"
      "choose from many restaurants,"
      or "there are many options nearby."
    - Avoid weak filler phrases such as:
      "if time allows,"
      "something for everyone,"
      or "choose from."

    Theme Prioritization Rules:
    - Prioritize recommendations that best match the user's requested theme.
    - For food-focused questions, prioritize restaurants, food halls,
      markets, cafes, bakeries, bars, and local dining experiences.
    - For nightlife-focused questions, prioritize bars, lounges, rooftop
      venues, breweries, entertainment districts, live music, comedy clubs,
      and evening activities.
    - For family-friendly questions, prioritize museums, parks, aquariums,
      outdoor activities, educational attractions, and interactive
      experiences.
    - For outdoor-focused questions, prioritize beaches, parks, trails,
      gardens, scenic overlooks, walking areas, and waterfront attractions.

    Itinerary Rules:
    - For itinerary-style questions, organize recommendations logically by
      day, time, area, or activity type.
    - If the user asks for a weekend, multi-day trip, or travel plan,
      structure the response with sections such as:
      Friday Night
      Saturday Morning
      Saturday Afternoon
      Saturday Night
      Sunday Morning
    - Build practical activity sequences that make geographic and thematic
      sense.
    - Avoid repeating the same attraction multiple times within an itinerary.
    - Mix categories naturally when appropriate, including food, nightlife,
      sightseeing, outdoor activities, museums, and neighborhoods.

    Formatting Rules:
    - Use clean spacing and readable formatting.
    - Use bullet points or sections when appropriate.
    - Keep the tone conversational, natural, and travel-oriented.
    - Do not mention chunk numbers or retrieval details directly.

    Output Format:
    1. Begin with a short direct response to the user's question.
    2. Then provide organized recommendations or itinerary sections.
    3. For each recommendation include:
       - The attraction, neighborhood, restaurant, or activity name
       - Why it is relevant to the request
       - Supporting citations such as (Passage 1)
    4. Only mention limited context if the retrieved passages truly do not
       contain enough information to answer the question.
    """

    user_prompt = f"""
    Retrieved travel passages:
    {context}

    User question:
    {query}

    Write a grounded travel answer.

    Important:
    - Use only the retrieved passages.
    - Do not repeat the same recommendation unless different passages add new information.
    - If multiple passages mention similar things, combine them into one stronger recommendation.
    - Cite support like: (Passage 1), (Passage 2).
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.45,
        max_tokens=512
    )

    return response.choices[0].message.content


# INGEST MODE

def run_ingest(
        data_dir: str,
        db_path: str,
        collection_name: str,
        embed_model: str,
        chunk_size: int,
        stride: int
) -> None:
    """
    Ingest documents: chunk, embed, and upsert to ChromaDB.

    Args:
        data_dir: Directory containing .txt files.
        db_path: Path for ChromaDB persistence.
        collection_name: Name of the collection.
        embed_model: OpenAI embedding model name.
        chunk_size: Characters per chunk.
        stride: Characters to advance between chunks.
    """
    print(f"Loading documents from {data_dir}...")
    documents = load_documents(data_dir)
    print(f"Found {len(documents)} document(s)")

    # Initialize clients
    openai_client = get_openai_client()
    chroma_client = get_chroma_client(db_path)
    collection = get_or_create_collection(chroma_client, collection_name)

    total_chunks = 0

    for doc in documents:
        filename = doc['filename']
        content = doc['content']

        print(f"\nProcessing: {filename}")

        # Chunk the document
        chunks = chunk_text(content, chunk_size, stride)
        print(f"  Created {len(chunks)} chunks")

        if not chunks:
            continue

        # Generate embeddings (batch for efficiency)
        print(f"  Generating embeddings...")
        embeddings = []
        batch_size = 1000

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            batch_embeddings = get_embeddings(openai_client, batch, embed_model)
            embeddings.extend(batch_embeddings)

        # Create IDs and metadata
        ids = [f"{filename}_{i}" for i in range(len(chunks))]

        city = infer_city(filename)
        metadatas = [
            {"source": filename, "city": city, "chunk_index": i}
            for i in range(len(chunks))
        ]

        # Upsert to ChromaDB
        print(f"  Upserting to ChromaDB...")
        upsert_chunks(collection, chunks, embeddings, metadatas, ids)

        total_chunks += len(chunks)

    print(f"\nIngestion complete! Total chunks: {total_chunks}")
    print(f"Collection '{collection_name}' now has {collection.count()} documents")


# SEARCH MODE

def run_search(
        query: str,
        top_k: int,
        db_path: str,
        collection_name: str,
        embed_model: str,
        alpha: float = 0.5
) -> List[Tuple[str, float, str]]:
    """
    Search the knowledge base using hybrid retrieval only.

    Args:
        query: Search query string.
        top_k: Number of results to return.
        db_path: Path to ChromaDB.
        collection_name: Name of the collection.
        embed_model: Embedding model for vector search.
        alpha: Weight for hybrid fusion.

    Returns:
        List of hybrid search results.
    """
    # Initialize ChromaDB
    chroma_client = get_chroma_client(db_path)
    collection = get_or_create_collection(chroma_client, collection_name)

    # Get all documents for BM25 and city filtering
    all_docs = collection.get(include=["documents", "metadatas"])
    documents = all_docs['documents'] if all_docs['documents'] else []
    doc_ids = all_docs['ids'] if all_docs['ids'] else []
    metadatas = all_docs['metadatas'] if all_docs['metadatas'] else []

    city_filter = detect_city_from_query(query)

    if city_filter:
        filtered = [
            (doc, meta, doc_id)
            for doc, meta, doc_id in zip(documents, metadatas, doc_ids)
            if meta.get("city") == city_filter
        ]

        if filtered:
            documents = [item[0] for item in filtered]
            metadatas = [item[1] for item in filtered]
            doc_ids = [item[2] for item in filtered]

    if not documents:
        print("No documents found in collection")
        return []

    # Get BM25 results
    bm25_index = build_bm25_index(documents)
    bm25_results = bm25_search(bm25_index, query, documents, top_k * 2)

    # Get vector results
    openai_client = get_openai_client()
    query_embedding = get_embeddings(openai_client, [query], embed_model)[0]
    vector_results = vector_search(collection, query_embedding, top_k * 2, city_filter)

    # Fuse results
    results = hybrid_fusion(bm25_results, vector_results, alpha)[:top_k]

    return results


# ANSWER MODE

def run_answer(
    query: str,
    top_k: int = 8,
    db_path: str = "./kb",
    collection_name: str = "travel",
    embed_model: str = "text-embedding-3-small",
    chat_model: str = "gpt-4o-mini",
    alpha: float = 0.5
) -> str:
    """
    Retrieve relevant chunks and generate a grounded answer.

    Args:
        query: User's question.
        top_k: Number of chunks to retrieve.
        db_path: Path to ChromaDB.
        collection_name: Collection name.
        embed_model: Embedding model name.
        chat_model: Chat model for answer generation.
        alpha: Hybrid fusion weight.

    Returns:
        Generated answer string.
    """
    # Retrieve relevant chunks
    results = run_search(
        query, top_k, db_path, collection_name, embed_model, alpha
    )

    # Extract text from results
    chunks = [r[2] for r in results]  # (id, score, text)

    # Format context
    chunks = remove_duplicate_chunks(chunks)
    context = format_context(chunks)

    # Generate answer
    print("\nGenerating grounded answer...")
    openai_client = get_openai_client()
    answer = generate_grounded_answer(openai_client, query, context, chat_model)

    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(answer)

    return answer


# MAIN ENTRY POINT

def main():
    """
    Main entry point for the RAG pipeline.
    Parses command line arguments and executes the appropriate mode.
    """
    parser = argparse.ArgumentParser(
        description="RAG Travel Assistant"
    )

    # Mode selection
    parser.add_argument(
        'mode',
        type=str,
        choices=['ingest', 'answer'],
        help="Mode to run: ingest or answer"
    )

    # data and database paths
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/corpus',
        help="Directory containing .txt files for ingestion"
    )

    parser.add_argument(
        '--db_path',
        type=str,
        default='./kb',
        help="Path for ChromaDB persistence"
    )

    parser.add_argument(
        '--collection',
        type=str,
        default='travel',
        help="ChromaDB collection name"
    )

    # Embedding model
    parser.add_argument(
        '--embed_model',
        type=str,
        default='text-embedding-3-small',
        help="OpenAI embedding model name"
    )

    # Chunking parameters
    parser.add_argument(
        '--size',
        type=int,
        default=400,
        help="Chunk size in characters"
    )

    parser.add_argument(
        '--stride',
        type=int,
        default=120,
        help="Stride between chunks (overlap = size - stride)"
    )

    # Search parameters
    parser.add_argument(
        '--query',
        type=str,
        help="Search query (required for search and answer modes)"
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=8,
        help="Number of results to retrieve"
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help="Hybrid fusion weight for vector scores (0-1)"
    )

    # Chat model for answer generation
    parser.add_argument(
        '--chat_model',
        type=str,
        default='gpt-4o-mini',
        help="Chat model for answer generation"
    )

    args = parser.parse_args()

    # Execute appropriate mode
    if args.mode == 'ingest':
        run_ingest(
            data_dir=args.data_dir,
            db_path=args.db_path,
            collection_name=args.collection,
            embed_model=args.embed_model,
            chunk_size=args.size,
            stride=args.stride
        )

    elif args.mode == 'answer':
        if not args.query:
            parser.error("--query is required for answer mode")

        run_answer(
            query=args.query,
            top_k=args.top_k,
            db_path=args.db_path,
            collection_name=args.collection,
            embed_model=args.embed_model,
            chat_model=args.chat_model,
            alpha=args.alpha
        )


if __name__ == "__main__":
    main()
