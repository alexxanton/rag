from .indexing import build_index
from .searching import search_query


class CLI:
    """Contains methods for the command line interface."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest data/raw/ and build the index under data/processed/."""
        build_index(max_chunk_size)

    def search(self, query: str, k: int) -> None:
        """Return the top-k sources for a single query."""
        search_query(query, k)

    def answer(self, query: str, k: int) -> None:
        """Answer a single query using the retrieved context."""
        pass

    def search_dataset(
        self, dataset_path: str, k: int, save_directory: str
    ) -> None:
        """
        Run search over a whole dataset and
        write a StudentSearchResults JSON file.
        """
        pass

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """
        Generate answers for a dataset,
        producing a StudentSearchResultsAndAnswer JSON file.
        """
        pass

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """
        Report your own recall@k against a ground-truth dataset,
        for your own testing."""
        pass
