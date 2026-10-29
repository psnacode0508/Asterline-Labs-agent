"""Local Chroma indexing and retrieval for immutable Asterline documentation."""

import json
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


CORPUS_PATH = Path("corpus.jsonl")
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "asterline_docs"
EXPECTED_CHUNK_COUNT = 75
METADATA_FIELDS = (
    "chunk_id",
    "doc_id",
    "title",
    "category",
    "owner",
    "source_url",
    "published",
    "version",
)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create CPU-only embeddings that run locally without an embedding API."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def load_corpus() -> List[Document]:
    """Parse the immutable JSONL corpus into LangChain documents."""
    if not CORPUS_PATH.is_file():
        raise FileNotFoundError(
            f"Corpus file not found: {CORPUS_PATH.resolve()}. "
            "Add the immutable corpus.jsonl file before indexing."
        )

    documents: List[Document] = []
    with CORPUS_PATH.open("r", encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                raise ValueError(f"Corpus contains an empty line at line {line_number}.")

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Corpus line {line_number} is not valid JSON."
                ) from error

            missing_fields = [
                field for field in ("text", *METADATA_FIELDS) if field not in record
            ]
            if missing_fields:
                raise ValueError(
                    f"Corpus line {line_number} is missing fields: "
                    f"{', '.join(missing_fields)}."
                )

            documents.append(
                Document(
                    page_content=record["text"],
                    metadata={field: record[field] for field in METADATA_FIELDS},
                )
            )

    if len(documents) != EXPECTED_CHUNK_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CHUNK_COUNT} corpus chunks, found {len(documents)}."
        )

    return documents


def _open_vector_store(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Open the configured local Chroma collection."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )


def get_vector_store() -> Chroma:
    """Return the existing complete index or build it from the immutable corpus."""
    embeddings = get_embeddings()
    vector_store = _open_vector_store(embeddings)

    if vector_store._collection.count() == EXPECTED_CHUNK_COUNT:
        return vector_store

    documents = load_corpus()
    vector_store.delete_collection()
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        ids=[document.metadata["chunk_id"] for document in documents],
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
    )


def get_retriever(k: int = 6):
    """Return a similarity retriever over the local Asterline documentation."""
    return get_vector_store().as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    store = get_vector_store()
    print(f"Total indexed chunks: {store._collection.count()}")

    results = get_retriever(k=2).invoke("on-call compensation")
    print("Top results for 'on-call compensation':")
    for document in results:
        print(f"- {document.metadata['chunk_id']}: {document.metadata['title']}")
