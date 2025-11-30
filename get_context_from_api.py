import re
import time
from serpapi import GoogleSearch


def clean_query(raw_query: str) -> str:
    """
    Simplify a natural-language query into a keyword-style phrase
    suitable for Google Trends input.
    """
    cleaned = re.sub(
        r'\b(what|who|where|when|why|tell me|show me|about|the|a|an|is|are)\b',
        '',
        raw_query,
        flags=re.I
    )
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned).strip()
    return cleaned


def infer_context_from_trends(query: str, api_key: str, geo: str = "US") -> dict:
    simple_query = clean_query(query)

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
    try:
        data = search.get_dict()
        time.sleep(1)
    except Exception as e:
        result["context"] = f"SERP API error: {str(e)}"
        return result

    if "error" in data:
        result["context"] = f"SERP API error: {data['error']}"
        return result

    related = data.get("related_queries", {})

    if "top" in related and isinstance(related["top"], list):
        result["top_related"] = [item["query"] for item in related["top"] if "query" in item][:5]

    if "rising" in related and isinstance(related["rising"], list):
        result["rising_related"] = [item["query"] for item in related["rising"] if "query" in item][:5]

    # Only keep Rising related in context
    parts = []
    if result["rising_related"]:
        parts.append("Rising related: " + ", ".join(result["rising_related"]))

    result["context"] = " ".join(parts) if parts else "No related topics found."
    return result



def get_context_for_query(api_key: str, geo: str = "US") -> dict:
    """
    Ask the user for a query and return the context dictionary.
    """
    query = input("Enter your query: ").strip()
    return infer_context_from_trends(query, api_key, geo)


# ---------- Example usage ----------
if __name__ == "__main__":
    API_KEY = "bbc5ace3aed5f02bcbd6affac66a496ed78fb1dd31f96cdbfffdcabf39bb0d0a"

    context_result = get_context_for_query(API_KEY)
    print("\nOriginal Query:", context_result["original_query"])
    print("Cleaned Query :", context_result["cleaned_query"])
    print("Top Related   :", context_result["top_related"])
    print("Rising Related:", context_result["rising_related"])
    print("Context       :", context_result["context"])
