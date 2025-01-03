from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class Chunk(BaseModel):
    doc_id: int
    chunk_id: int
    body: str
    embedding: list[float] = []
    summary: str=""

class Document(BaseModel):
    doc_id: int = -1
    doc_type: str=""
    doc_title: str = ""
    doc_source: str = ""
    chunk_list: list[Chunk] = []
    raw_text: str=""


# gpt : structured output
class str_struct(BaseModel):
    output: str

class intlist_struct(BaseModel):
    output: list[int]