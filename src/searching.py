import json
import bm25s
from pathlib import Path
from .globals import PROCESSED_PATH
from .data_models import (
    RagDataset,
    UnansweredQuestion,
    AnsweredQuestion,
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults
)


def init_retriever():
    """Initialize the retriever."""
    chunks_file = Path(PROCESSED_PATH) / "chunks.json"
    chunks = json.loads(chunks_file.read_text())

    retriever = bm25s.BM25.load(
        PROCESSED_PATH,
        load_corpus=True
    )
    return (chunks, retriever)


def execute_query(retriever, query: str, k: int):
    """Tokenizes a query and retrieves from the indexing."""
    tokens = bm25s.tokenize(query)

    results, scores = retriever.retrieve(
        tokens,
        k=k
    )
    return results


def search_query(query: str, k: int) -> None:
    """Perform a single query and display its results."""
    chunks, retriever = init_retriever()
    results = execute_query(retriever, query, k)

    for chunk_id in results[0]:
        chunk = chunks[chunk_id]
        start_index = chunk["start_index"]

        #print(score)
        print(chunk["file_path"], f"[{start_index}:{len(chunk['text'])}]")


def search_queries(dataset_path: str, k: int, save_directory: str) -> None:
    """Perform multiple queries and store the output."""
    save = Path(save_directory) / "dataset_docs_public.json"
    dataset = RagDataset.model_validate_json(Path(dataset_path).read_text())
    results_list = []

    chunks, retriever = init_retriever()
    for query in dataset.rag_questions:
        results = execute_query(retriever, query.question, k)
        results_list.append(
            MinimalSearchResults(
                question=query.question,
                question_id=query.question_id,
                retrieved_sources=[
                    MinimalSource.model_validate(chunks[idx])
                    for idx in results[0]
                ]
            )
        )

    search_results = StudentSearchResults(search_results=results_list, k=k)
    dump = json.dumps(search_results.model_dump(), indent=2)
    #print(dump)
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(dump)
