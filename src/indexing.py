import json
import bm25s
from tqdm import tqdm
from pathlib import Path
from langchain_text_splitters import PythonCodeTextSplitter
from .globals import DOCS_PATH, PROCESSED_PATH


def split_file(file):
    """Split file into chunks."""
    text = file.read_text()

    splitter = PythonCodeTextSplitter(
        chunk_size=2000,
        chunk_overlap=0,
        add_start_index=True
    )

    docs = splitter.create_documents(
        [text],
        metadatas=[{"file_path": str(file)}]
    )
    return docs


def create_chunks():
    chunks = []
    root = DOCS_PATH
    files = [
        path
        for path in Path(root).rglob("*")
        if path.suffix in {".py", ".md"}
    ]

    for file in tqdm(files, desc="Chunking", unit="file"):
        chunks.extend(split_file(file))

    return chunks


def save_chunks(chunks) -> None:
    chunks_file = Path(PROCESSED_PATH) / "chunks.json"

    data = [
        {
            #"text": chunk.page_content,
            "file_path": chunk.metadata["file_path"],
            "first_character_index": chunk.metadata["start_index"],
            "last_character_index": (
                chunk.metadata["start_index"] + len(chunk.page_content) - 1
            )
        }
        for chunk in chunks
    ]
    chunks_file.parent.mkdir(parents=True, exist_ok=True)
    chunks_file.write_text(json.dumps(data))

def build_index(max_chunk_size: int) -> None:
    index_path = Path(PROCESSED_PATH)

    chunks = create_chunks()
    corpus = [
        chunk.page_content for chunk in chunks
    ]

    tokens = bm25s.tokenize(corpus)
    retriever = bm25s.BM25()
    retriever.index(tokens)

    index_path.mkdir(exist_ok=True)
    save_chunks(chunks)

    retriever.save(str(index_path))
    bm25s.tokenize(corpus, return_ids=True)
