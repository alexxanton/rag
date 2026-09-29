import json
import bm25s
from tqdm import tqdm
from pathlib import Path
from langchain_text_splitters import PythonCodeTextSplitter


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
    root = "vllm-0.10.1/"
    files = [
        path
        for path in Path(root).rglob("*")
        if path.suffix in {".py", ".md"}
    ]

    for file in tqdm(files, desc="Chunking", unit="file"):
        chunks.extend(split_file(file))

    return chunks


def save_chunks(chunks):
    chunks_file = Path("index/chunks.json")

    data = [
        {
            "text": chunk.page_content,
            "file_path": chunk.metadata["file_path"],
            "start_index": chunk.metadata["start_index"],
        }
        for chunk in chunks
    ]
    chunks_file.parent.mkdir(parents=True, exist_ok=True)
    chunks_file.write_text(json.dumps(data))

def build_index(max_chunk_size: int):
    index_path = Path("index")

    chunks = create_chunks()
    corpus = [
        chunk.page_content for chunk in chunks
    ]

    tokens = bm25s.tokenize(corpus)
    retriever = bm25s.BM25()
    retriever.index(tokens)

    index_path.mkdir(exist_ok=True)
    save_chunks(chunks)

    retriever.save(str(index_path / "bm25"))
    bm25s.tokenize(corpus, return_ids=True)
