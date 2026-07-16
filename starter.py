"""Starter code for the monitoring homework.

Sets up the text-search RAG from homework 1 and a shared OpenAI client.
"""

from openai import OpenAI

from gitsource import GithubRepositoryDataReader
from minsearch import Index

from rag_helper import RAGBase
import os
import json
from dotenv import load_dotenv

load_dotenv()

COMMIT = "8c1834d"

# --- Load the course lessons (same as HW1, HW2, HW4) ---
cache_file = "documents_cache.json"
if os.path.exists(cache_file):
    with open(cache_file, "r") as f:
        documents = json.load(f)
else:
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id=COMMIT,
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    documents = [file.parse() for file in reader.read()]
    with open(cache_file, "w") as f:
        json.dump(documents, f)

index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(documents)

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))
rag = RAGBase(index=index, llm_client=client)

if __name__ == "__main__":
    query = "How does the agentic loop keep calling the model until it stops?"
    answer = rag.rag(query)
    print(answer)

