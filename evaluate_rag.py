"""
Evaluates the RAG Travel Assistant using Precision@K, query city detection
accuracy, retrieved chunk city accuracy, source diversity, and latency.

Author: Sebastian Escobar-Mesa
Date: 5-9-2026
"""

import csv
import time
from typing import Dict, List

from travel_rag import (
    run_search,
    run_answer,
    detect_city_from_query,
    get_chroma_client,
    get_or_create_collection
)


# Test queries used to test retrieval quality, city filtering accuracy, source diversity, and response generation.
TEST_QUERIES = [
    {
        "query": "What are some good nightlife options in New Orleans?",
        "expected_city": "New Orleans",
        "expected_keywords": [
            "bar", "music", "nightlife", "bourbon", "frenchmen"
        ]
    },
    {
        "query": "Plan a weekend trip to New York City.",
        "expected_city": "New York City",
        "expected_keywords": [
            "new york", "museum", "central park", "restaurant", "itinerary"
        ]
    },
    {
        "query": "What are family friendly activities in Atlanta?",
        "expected_city": "Atlanta",
        "expected_keywords": [
            "family", "museum", "park", "aquarium", "children"
        ]
    },
    {
        "query": "What are outdoor things to do in Los Angeles?",
        "expected_city": "Los Angeles",
        "expected_keywords": [
            "beach", "park", "outdoor", "hike", "walk"
        ]
    },
    {
        "query": "Where should I eat in Houston?",
        "expected_city": "Houston",
        "expected_keywords": [
            "restaurant", "food", "eat", "dining", "market"
        ]
    },
    {
        "query": "Give me museum recommendations in New York City.",
        "expected_city": "New York City",
        "expected_keywords": [
            "museum", "art", "history", "gallery"
        ]
    },
    {
        "query": "What should I do in New Orleans for a short weekend?",
        "expected_city": "New Orleans",
        "expected_keywords": [
            "french quarter", "music", "food", "itinerary", "garden"
        ]
    },
    {
        "query": "What are free things to do in Los Angeles?",
        "expected_city": "Los Angeles",
        "expected_keywords": [
            "free", "beach", "park", "walk", "museum"
        ]
    },
    {
        "query": "What are good tourist attractions in Atlanta?",
        "expected_city": "Atlanta",
        "expected_keywords": [
            "attraction", "museum", "park", "aquarium", "tour"
        ]
    },
    {
        "query": "What are good places to visit in Houston?",
        "expected_city": "Houston",
        "expected_keywords": [
            "museum", "park", "space", "market", "downtown"
        ]
    },
]


def keyword_relevance_score(
        text: str,
        expected_keywords: List[str]
) -> bool:
    """
    Check whether a retrieved chunk is relevant by keyword matching.

    Args:
        text: Retrieved chunk text.
        expected_keywords: Keywords expected in relevant chunks.

    Returns:
        True if at least one expected keyword is found, otherwise False.
    """
    text_lower = text.lower()

    for keyword in expected_keywords:
        if keyword.lower() in text_lower:
            return True

    return False


def build_metadata_lookup(
        db_path: str,
        collection_name: str
) -> Dict[str, Dict]:
    """
    Build a lookup table from chunk text to ChromaDB metadata.

    Args:
        db_path: Path to the ChromaDB database.
        collection_name: Name of the ChromaDB collection.

    Returns:
        Dictionary mapping chunk text to its metadata.
    """
    chroma_client = get_chroma_client(db_path)
    collection = get_or_create_collection(chroma_client, collection_name)
    stored_data = collection.get(include=["documents", "metadatas"])

    documents = stored_data["documents"] if stored_data["documents"] else []
    metadatas = stored_data["metadatas"] if stored_data["metadatas"] else []

    # Store metadata for each chunk
    return {
        document: metadata
        for document, metadata in zip(documents, metadatas)
    }


def evaluate_query(
        test_case: Dict,
        metadata_lookup: Dict[str, Dict],
        top_k: int = 8,
        alpha: float = 0.5
) -> Dict:
    """
    Evaluate a single test query against the RAG system.

    Measures:
    - Precision@K
    - Query city detection accuracy
    - Retrieved chunk city accuracy
    - Source diversity
    - Retrieval latency
    - Response generation latency

    Args:
        test_case: Dictionary containing query, expected_city,
            and expected_keywords.
        metadata_lookup: Dictionary mapping chunk text to metadata.
        top_k: Number of retrieved chunks.
        alpha: Hybrid retrieval fusion weight.

    Returns:
        Dictionary containing evaluation metrics and generated answer.
    """
    query = test_case["query"]
    expected_city = test_case["expected_city"]
    expected_keywords = test_case["expected_keywords"]

    print("=" * 80)
    print(f"Query: {query}")
    print(f"Expected city: {expected_city}")

    # Measure retrieval latency
    retrieval_start = time.time()

    results = run_search(
        query=query,
        top_k=top_k,
        db_path="./kb",
        collection_name="travel",
        embed_model="text-embedding-3-small",
        alpha=alpha
    )

    retrieval_time = time.time() - retrieval_start

    relevant_count = 0
    correct_city_count = 0
    source_names = set()

    # Evaluate every retrieved chunk for keyword relevance, city correctness, and source diversity.
    for _, _, text in results:
        if keyword_relevance_score(text, expected_keywords):
            relevant_count += 1

        metadata = metadata_lookup.get(text, {})
        chunk_city = metadata.get("city")
        source_name = metadata.get("source")

        if chunk_city == expected_city:
            correct_city_count += 1

        if source_name:
            source_names.add(source_name)

    precision_at_k = relevant_count / top_k if top_k > 0 else 0
    chunk_city_accuracy = correct_city_count / top_k if top_k > 0 else 0
    source_diversity = len(source_names)

    # Verify that the query parser detected correct supported city.
    detected_city = detect_city_from_query(query)
    query_city_detected_correctly = detected_city == expected_city

    # Measure response generation time
    response_start = time.time()

    answer = run_answer(
        query=query,
        top_k=top_k,
        db_path="./kb",
        collection_name="travel",
        embed_model="text-embedding-3-small",
        chat_model="gpt-4o-mini",
        alpha=alpha
    )

    response_time = time.time() - response_start

    print(f"Precision@{top_k}: {precision_at_k:.3f}")
    print(f"Query city detected correctly: {query_city_detected_correctly}")
    print(f"Retrieved chunk city accuracy: {chunk_city_accuracy:.3f}")
    print(f"Source diversity: {source_diversity}")
    print(f"Retrieval time: {retrieval_time:.2f} seconds")
    print(f"Response time: {response_time:.2f} seconds")
    print()

    return {
        "query": query,
        "expected_city": expected_city,
        "detected_city": detected_city,
        "query_city_detected_correctly": query_city_detected_correctly,
        f"precision_at_{top_k}": precision_at_k,
        "chunk_city_accuracy": chunk_city_accuracy,
        "source_diversity": source_diversity,
        "retrieval_time_seconds": retrieval_time,
        "response_time_seconds": response_time,
        "answer": answer
    }


def run_evaluation(
        top_k: int = 8,
        alpha: float = 0.5
) -> None:
    """
    Run the full evaluation pipeline for all test queries.

    Args:
        top_k: Number of retrieved chunks used for evaluation.
        alpha: Hybrid retrieval fusion weight.

    Returns:
        None.
    """
    metadata_lookup = build_metadata_lookup(
        db_path="./kb",
        collection_name="travel"
    )

    all_results = []

    for test_case in TEST_QUERIES:
        result = evaluate_query(
            test_case=test_case,
            metadata_lookup=metadata_lookup,
            top_k=top_k,
            alpha=alpha
        )

        all_results.append(result)

    # Aggregate metrics across all evaluation queries.
    avg_precision = (
        sum(r[f"precision_at_{top_k}"] for r in all_results)
        / len(all_results)
    )

    query_city_accuracy = (
        sum(
            1
            for r in all_results
            if r["query_city_detected_correctly"]
        ) / len(all_results)
    )

    avg_chunk_city_accuracy = (
        sum(r["chunk_city_accuracy"] for r in all_results)
        / len(all_results)
    )

    avg_source_diversity = (
        sum(r["source_diversity"] for r in all_results)
        / len(all_results)
    )

    avg_retrieval_time = (
        sum(r["retrieval_time_seconds"] for r in all_results)
        / len(all_results)
    )

    avg_response_time = (
        sum(r["response_time_seconds"] for r in all_results)
        / len(all_results)
    )

    print("=" * 80)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Average Precision@{top_k}: {avg_precision:.3f}")
    print(f"Query City Detection Accuracy: {query_city_accuracy:.3f}")
    print(f"Average Retrieved Chunk City Accuracy: {avg_chunk_city_accuracy:.3f}")
    print(f"Average Source Diversity: {avg_source_diversity:.2f}")
    print(f"Average Retrieval Time: {avg_retrieval_time:.2f} seconds")
    print(f"Average Response Time: {avg_response_time:.2f} seconds")

    output_file = "rag_evaluation_results.csv"

    # Save detailed metrics and generated answers for report analysis.
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nSaved results to {output_file}")


if __name__ == "__main__":
    run_evaluation(top_k=8, alpha=0.5)