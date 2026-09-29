import json
import bm25s
from pathlib import Path
from .globals import PROCESSED_PATH


def search_query(query: str, k: int):
    chunks_file = Path(PROCESSED_PATH) / "chunks.json"
    documents = json.loads(chunks_file.read_text())

    retriever = bm25s.BM25.load(
        PROCESSED_PATH,
        load_corpus=True
    )

    tokens = bm25s.tokenize(query)

    results, scores = retriever.retrieve(
        tokens,
        k=k
    )

    for doc_id, score in zip(results[0], scores[0]):
        document = documents[doc_id]

        print(score)
        print(document)
