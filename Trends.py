import re
import time
from serpapi import GoogleSearch


def clean_query(raw_query: str):
    """
    Simplify a natural-language query into a keyword-style phrase
    suitable for Google Trends input.
    """
    cleaned = re.sub(
        r'\b(what|who|where|when|why|how|tell me|show me|about|the|a|an|is|are)\b',
        '',
        raw_query,
        flags=re.I
    )
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned).strip()
    return cleaned


def infer_context_from_trends(query, api_key, geo="US"):
    """
    Uses a cleaned version of the query for Trends,
    but returns full structured result: top_related, rising_related, context.
    """
    simple_query = clean_query(query)

    # Always return consistent dictionary
    result = {
        "original_query": query,
        "cleaned_query": simple_query,
        "top_related": [],
        "rising_related": [],
        "context": ""
    }

    if not simple_query:
        result["context"] = "Query too vague after cleaning."
        return result

    params = {
        "engine": "google_trends",
        "q": simple_query,
        "data_type": "RELATED_QUERIES",
        "geo": geo,
        "date": "today 5-y",
        "api_key": api_key
    }

    search = GoogleSearch(params)
    data = search.get_dict()
    time.sleep(1)

    # Handle API-side errors
    if "error" in data:
        print(f"[ERROR] SERP API says: {data['error']}")
        result["context"] = "No related topics (SERP API error)."
        return result

    # Extract trend information
    related = data.get("related_queries", {})

    if "top" in related and isinstance(related["top"], list):
        result["top_related"] = [
            item["query"] for item in related["top"] if "query" in item
        ][:5]

    if "rising" in related and isinstance(related["rising"], list):
        result["rising_related"] = [
            item["query"] for item in related["rising"] if "query" in item
        ][:5]

    # Build context string
    parts = []
    if result["top_related"]:
        parts.append("Top related: " + ", ".join(result["top_related"]))
    if result["rising_related"]:
        parts.append("Rising related: " + ", ".join(result["rising_related"]))

    result["context"] = " ".join(parts) if parts else "No related topics found."

    return result


# ---------- test ----------
if __name__ == "__main__":
    API_KEY = "bbc5ace3aed5f02bcbd6affac66a496ed78fb1dd31f96cdbfffdcabf39bb0d0a"

    queries = [
        "Tell me more about Tesla",
        "Who is Elon Musk",
        "When is the Apple event",
        "What about F1 race",
        "Who Microsoft Boss",
        "Who White nights novel"
    ]

    for q in queries:
        result = infer_context_from_trends(q, API_KEY)
        print("\nQuery:", result["original_query"])
        print("Cleaned:", result["cleaned_query"])
        print("Top   :", result["top_related"])
        print("Rising:", result["rising_related"])
        print("Context:", result["context"])
