from typing import Literal, NotRequired

from pydantic import TypeAdapter
from typing_extensions import TypedDict


class ContextChunk(TypedDict):
    type: Literal["previous", "current", "next"]
    content: str

class DocSearchResult(TypedDict):
    content: str
    source_file: str
    section_title: str | None
    line_start: int
    line_end: int
    doc_type: str
    score: float
    context: NotRequired[list[ContextChunk]]

class SearchDocsResponse(TypedDict):
    status: Literal["ok"]
    query: str
    results: list[DocSearchResult]
    count: int
    hint: NotRequired[str]

class ErrorResponse(TypedDict):
    error: Literal[True]
    error_type: str
    message: str
    details: str | dict | None

ToolOutput = SearchDocsResponse | ErrorResponse

adapter = TypeAdapter(ToolOutput)
try:
    dict_val = {
        "status": "ok",
        "query": "test",
        "count": 1,
        "results": [
            {
                "content": "2. Configure...",
                "source_file": "path/to/file",
                "section_title": None,
                "line_start": 1,
                "line_end": 10,
                "score": 0.0,
                "doc_type": "markdown",
            }
        ]
    }
    adapter.validate_python(dict_val)
    print("VALIDATION SUCCESS")
except Exception as e:
    print("ERROR:", e)
