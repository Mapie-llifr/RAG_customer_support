from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from config import (
    ELASTIC_HOST,
    ELASTIC_INDEX,
    EMBEDDING_MODEL_NAME
)
from llm import LocalLLM

LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def retrieve_context(question: str, top_k: int = 3) -> list[str]:
    es = Elasticsearch(ELASTIC_HOST)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    query_embedding = embedder.encode(question).tolist()

    response = es.search(
        index=ELASTIC_INDEX,
        knn={
            "field": "embedding",
            "query_vector": query_embedding,
            "k": top_k,
            "num_candidates": 10
        }
    )

    return [hit["_source"]["content"] for hit in response["hits"]["hits"]]


def build_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(contexts)

    prompt = f"""
    You are a customer support assistant.
    Answer the question using ONLY the context below.
    If the answer is not in the context, say:
    "Je n'ai pas assez d'information."

    Context:
    {context_block}

    User question (in French):
    {question}

    Final answer (in French):
    """
    return prompt.strip()


def rag_answer(question: str) -> str:
    contexts = retrieve_context(question)
    prompt = build_prompt(question, contexts)

    llm = LocalLLM(LLM_MODEL_NAME)
    return llm.generate(prompt)


if __name__ == "__main__":
    #question = "How can I reset my password?"
    question = "Faut-il manger des carottes ?"
    #question = "A quoi ça sert la secu ?"
    #question = "A quoi ça sert Mon Espace Santé?"
    answer = rag_answer(question)
    print("\nFinal answer:\n")
    print(answer)

