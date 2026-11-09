"""
This script implements a minimal Retrieval-Augmented Generation (RAG) pipeline. It retrieves relevant documents 
from a predefined knowledge base using TF-IDF vectorization and cosine similarity, then generates a contextual 
response to a user-provided question using a Flan-T5 language model. The script evaluates the quality of the 
response based on relevance, references, and length, providing a critique and confidence score. It can be run 
from the command line with options to specify the number of documents to retrieve and output format.
"""

# minimal rag

import sys
import numpy as np
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

MAX_NEW_TOKENS = 120
MIN_NEW_TOKENS = 15
MIN_SIMILARITY = 0.0
MODEL_NAME = "google/flan-t5-base"
DEFAULT_K = 3

DOCS = [
    {"id": "1", "txt": "Python is a high-level, interpreted programming language. It was created by Guido van Rossum and first released in 1991. Python emphasizes code readability with its notable use of significant whitespace."},
    {"id": "2", "txt": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+. It is based on standard Python type hints and provides automatic API documentation. FastAPI is one of the fastest Python frameworks available."},
    {"id": "3", "txt": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience. It uses algorithms to parse data, learn from it, and make predictions or decisions without being explicitly programmed."},
    {"id": "4", "txt": "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them to generate informed, contextual answers. This approach improves accuracy and reduces hallucinations."},
    {"id": "5", "txt": "Vector databases store data as high-dimensional vectors and enable efficient similarity search. They are essential for modern AI applications, particularly in semantic search and retrieval systems. Common examples include Pinecone, Weaviate, and Chroma."}
]


def parse_args(argv):
    """Return (question, k, use_json)."""
    if len(argv) < 2:
        print("usage: python script.py <question> [-k NUM] [--json]")
        print("example: python script.py 'What is RAG?' -k 2 --json")
        sys.exit(1)

    question = argv[1]
    k = DEFAULT_K
    use_json = False
    i = 2

    while i < len(argv):
        if argv[i] == "-k":
            k = int(argv[i+1])
            i += 2
        elif argv[i] == "--json":
            use_json = True
            i += 1
        else:
            i += 1

    return question, k, use_json


def load_model():
    try:
        print("loading model...")
        llm = pipeline("text2text-generation", model=MODEL_NAME)
        print("model loaded")
        return llm
    except Exception as e:
        print(f"failed to load model: {e}", file=sys.stderr)
        sys.exit(1)


def retrieve_docs(question, vectorizer, vectors, k):
    """Return up to k docs scoring above MIN_SIMILARITY, best match first."""
    query_vec = vectorizer.transform([question])
    similarities = cosine_similarity(query_vec, vectors)[0]
    indices = np.argsort(similarities)[-k:][::-1]

    retrieved = []  # fresh list per call, so repeated calls can't accumulate results
    for i in indices:
        if similarities[i] > MIN_SIMILARITY:
            retrieved.append({"doc": DOCS[i], "score": float(similarities[i])})
            # this break looks redundant but is kept on purpose: for k <= 0 the slice above returns extra docs and this caps it
            # proper k validation comes with argparse
            if len(retrieved) >= k:
                break
    return retrieved


def build_prompt(question, retrieved_docs):
    ctx = ""
    for r in retrieved_docs:
        ctx += f"[{r['doc']['id']}] {r['doc']['txt']}\n\n" # pass doc id in prompt instead of a counter so model's citations reference the right doc

    return (
        f"Context:\n{ctx}\nQuestion: {question}\n\n"
        "Answer the question based only on the context above. "
        "Be concise. Answer in complete sentences with correct punctuation. Reference citations by id (the numbers in brackets in the context) if needed.\n\n"
        "Answer:"
    )


def generate_answer(llm, prompt):
    output = llm(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=MIN_NEW_TOKENS,
        do_sample=False # make the model greedy and deterministic to avoid hallucinations
    )
    answer = output[0]["generated_text"].strip()
    return answer if answer else "idk"


def validate_output(answer, retrieved_docs):
    """Return (is_valid, score, problems)."""
    problems = []
    score = 1.0

    if "idk" in answer.lower():
        problems.append("no info")
        score = score * 0.5

    if len(answer) < 20:
        problems.append("short")
        score = score * 0.3

    refs = re.findall(r'\[(\d+)\]', answer)

    if len(retrieved_docs) > 0:
        if len(refs) == 0:
            problems.append("no refs")
            score = score * 0.7

        avg = sum(r["score"] for r in retrieved_docs) / len(retrieved_docs)

        if avg < 0.1:
            problems.append("low relevance")
            score = score * 0.6

    # same as the original nested ifs
    is_valid = score > 0.6 and len(problems) < 3
    return is_valid, score, problems


def print_report(question, answer, retrieved_docs, is_valid, score, problems, use_json):
    print("\nQ: {}".format(question))
    print(f"\nA: {answer}\n")

    if retrieved_docs:
        print("Citations:")
        for r in retrieved_docs:
            doc_text = r['doc']['txt']
            if len(doc_text) > 80:
                doc_text = doc_text[:80] + "..."
            print("  [{}] (score: {:.3f}) {}".format(r['doc']['id'], r['score'], doc_text))
    else:
        print("Citations: None")

    if is_valid:
        print("\nCritique: OK (confidence: {:.2f})".format(score))
    else:
        print(f"\nCritique: ISSUES (confidence: {score:.2f})")

    if len(problems) > 0:
        print("Problems: %s" % ', '.join(problems))

    if use_json:
        result = {
            "answer": answer,
            "citations": retrieved_docs,
            "critique": {"ok": is_valid, "score": score, "problems": problems}
        }
        print(json.dumps(result, indent=2))


def main():
    question, k, use_json = parse_args(sys.argv)

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([d["txt"] for d in DOCS])

    retrieved_docs = retrieve_docs(question, vectorizer, vectors, k)

    if not retrieved_docs:
        # no docs relevant to the question, no further work can be done, skip the model load and return 
        answer, is_valid, score, problems = "idk", False, 0.4, ["no docs"]
    else:
        llm = load_model()
        prompt = build_prompt(question, retrieved_docs)
        answer = generate_answer(llm, prompt)
        is_valid, score, problems = validate_output(answer, retrieved_docs)

    print_report(question, answer, retrieved_docs, is_valid, score, problems, use_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())