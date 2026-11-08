"""
This script implements a minimal Retrieval-Augmented Generation (RAG) pipeline. It retrieves relevant documents 
from a predefined knowledge base using TF-IDF vectorization and cosine similarity, then generates a contextual 
response to a user-provided question using a GPT-2 language model. The script evaluates the quality of the 
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

try:
    print("loading model...")
    llm = pipeline("text-generation", model="gpt2", max_new_tokens=256)
    print("model loaded")
except:
    pass

docs = [
    {"id": "1", "txt": "Python is a high-level, interpreted programming language. It was created by Guido van Rossum and first released in 1991. Python emphasizes code readability with its notable use of significant whitespace."},
    {"id": "2", "txt": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+. It is based on standard Python type hints and provides automatic API documentation. FastAPI is one of the fastest Python frameworks available."},
    {"id": "3", "txt": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience. It uses algorithms to parse data, learn from it, and make predictions or decisions without being explicitly programmed."},
    {"id": "4", "txt": "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them to generate informed, contextual answers. This approach improves accuracy and reduces hallucinations."},
    {"id": "5", "txt": "Vector databases store data as high-dimensional vectors and enable efficient similarity search. They are essential for modern AI applications, particularly in semantic search and retrieval systems. Common examples include Pinecone, Weaviate, and Chroma."}
]

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([x["txt"] for x in docs])

retrieved_docs = []
answer = ""
problems = []
score = 1.0
is_valid = False
temp_storage = {}

if len(sys.argv) < 2:
    print("usage: python minimal-rag.py <question> [-k NUM] [--json]")
    print("example: python minimal-rag.py 'What is RAG?' -k 2 --json")
    sys.exit(1)

question = sys.argv[1]
k = 3
use_json = False
i = 2

while i < len(sys.argv):
    if sys.argv[i] == "-k":
        k = int(sys.argv[i+1])
        i += 2
    elif sys.argv[i] == "--json":
        use_json = True
        i += 1
    else:
        i += 1

retrieved_docs = []
answer = ""
problems = []
score = 1.0
temp_storage["question"] = question

def get_data():
    global retrieved_docs, temp_storage
    query_vec = vectorizer.transform([question])
    similarities = cosine_similarity(query_vec, vectors)[0]
    indices = np.argsort(similarities)[-k:][::-1]
    temp_storage["sims"] = similarities

    for i in indices:
        if similarities[i] > 0:
            retrieved_docs.append({"doc": docs[i], "score": float(similarities[i])})
            if len(retrieved_docs) >= k:
                break

get_data()

if len(retrieved_docs) == 0:
    answer = "idk"
    problems.append("no docs")
    score = score * 0.4
    is_valid = False
    print("\nQ: {}".format(question))
    print(f"\nA: {answer}\n")
    print("Citations: None")
    print("\nCritique: ISSUES (confidence: {:.2f})".format(score))
    print("Problems: %s" % ', '.join(problems))
    if use_json:
        print(json.dumps({"answer": answer, "citations": retrieved_docs, "critique": {"ok": is_valid, "score": score, "problems": problems}}, indent=2))
    sys.exit(0)

def call_model():
    global answer, temp_storage
    ctx = ""
    counter = 0
    for r in retrieved_docs:
        counter += 1
        ctx += f"[{counter}] {r['doc']['txt']}\n\n"

    temp_storage["context"] = ctx
    prompt = "Context:\n{}\nQuestion: {}\n\nAnswer the question based only on the context above. Cite sources using [1], [2], etc. Be concise.\n\nAnswer:".format(ctx, temp_storage['question'])
    output = llm(prompt)
    answer = output[0]['generated_text'].split("Answer:")[-1].strip()

    if answer == "":
        answer = "idk"

call_model()

def validate_output():
    global problems, score, is_valid
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

        avg = 0
        for r in retrieved_docs:
            avg += r["score"]
        avg = avg / len(retrieved_docs)

        if avg < 0.1:
            problems.append("low relevance")
            score = score * 0.6

    if score > 0.6:
        if len(problems) < 3:
            is_valid = True
        else:
            is_valid = False
    else:
        is_valid = False

validate_output()

print("\nQ: {}".format(question))
print(f"\nA: {answer}\n")
print("Citations:")

for r in retrieved_docs:
    doc_text = r['doc']['txt']
    if len(doc_text) > 80:
        doc_text = doc_text[:80] + "..."
    print("  [{}] (score: {:.3f}) {}".format(r['doc']['id'], r['score'], doc_text))

if is_valid:
    print("\nCritique: OK (confidence: {:.2f})".format(score))
else:
    print(f"\nCritique: ISSUES (confidence: {score:.2f})")

if len(problems) > 0:
    print("Problems: %s" % ', '.join(problems))

result = {
    "answer": answer,
    "citations": retrieved_docs,
    "critique": {"ok": is_valid, "score": score, "problems": problems}
}

if use_json:
    print(json.dumps(result, indent=2))