import uuid
from typing import List
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Represents a single source of information."""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represents an unanswered question."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represents an answered question."""
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represents a dataset of RAG questions."""
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Represents the search results."""
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represents the search results and an answer."""
    answer: str


class StudentSearchResults(BaseModel):
    """Represents search results."""
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Represents search results and search results with answers."""
    search_results: List[MinimalAnswer]
    k: int
