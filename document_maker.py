from dataclasses import dataclass, field
from typing import Optional, Required
from llama_index.core import Document

@dataclass
class DocumentMaker:
  """Creates Document objects with consistent metadata."""
  
  id: Required[str] = field(metadata={"description": "Unique identifier for the document"})
  page_label: Optional[str] = None
  file_name: Optional[str] = None
  file_type: Optional[str] = None
  creation_date: Optional[str] = None

  def get_metadata(self) -> dict:
    """Returns all non-empty metadata fields."""
    return {k: v for k, v in self.__dict__.items() if v is not None and v != ""}
  
  def set_metadata(self, metadata: dict) -> None:
    """Updates metadata fields from a dictionary."""
    for k, v in metadata.items():
      if hasattr(self, k):  # Only set known attributes
        setattr(self, k, v)
      else:
        raise AttributeError(f"'{k}' is not a valid metadata field")
  
  def create_document(self, text: str) -> Document:
    """Creates a Document with the current metadata."""
    if not text:
      raise ValueError("Document text cannot be empty")
    return Document(text=text, metadata=self.get_metadata())
