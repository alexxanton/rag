from .indexing import build_index
from .searching import search_query


class CLI:
    """Contains methods for the command line interface."""
    def index(self, p=""):
        build_index()

    def search(self):
        search_query()
