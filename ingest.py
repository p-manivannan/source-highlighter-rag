"""Ingest PDFs from ./data into a persistent local ChromaDB collection.

Run this script once after placing PDF files in the data directory:

    python ingest.py

The script is safe to re-run for normal use. It parses and chunks all PDFs
before replacing the existing named Chroma collection.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "enterprise_rag"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# A 400-token target is inside the requested 300-500 token range.
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


def load_and_chunk_pdfs(pdf_paths: list[Path]):
    """Read every PDF and return LlamaIndex nodes with citation metadata."""
    pdf_reader = PDFReader()
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    all_nodes = []

    for pdf_path in pdf_paths:
        LOGGER.info("Reading %s", pdf_path.name)

        try:
            # PDFReader usually returns one Document per PDF page.
            documents = pdf_reader.load_data(file=pdf_path)
            nodes = splitter.get_nodes_from_documents(documents)
        except Exception as exc:
            # One damaged PDF should not prevent other valid PDFs from loading.
            LOGGER.error("Skipping %s because it could not be read: %s", pdf_path.name, exc)
            continue

        if not nodes:
            LOGGER.warning("No extractable text was found in %s; skipping it.", pdf_path.name)
            continue

        # Chunk IDs begin at 1 because that is friendlier in user-facing citations.
        # The two required fields are attached after splitting, so EVERY saved node
        # has an exact source-file/chunk mapping.
        for chunk_id, node in enumerate(nodes, start=1):
            node.metadata["source_file"] = pdf_path.name
            node.metadata["chunk_id"] = chunk_id
            try:
                node.metadata["page_number"] = max(
                    1, int(node.metadata.get("page_label", 1))
                )
            except (TypeError, ValueError):
                node.metadata["page_number"] = 1

            # Keep citation metadata in Chroma, but do not let a filename or chunk
            # number influence the semantic embedding.
            node.excluded_embed_metadata_keys = [
                "source_file",
                "chunk_id",
                "page_number",
            ]

        all_nodes.extend(nodes)
        LOGGER.info("Created %d chunks from %s", len(nodes), pdf_path.name)

    return all_nodes


def create_local_embedding_model() -> HuggingFaceEmbedding:
    """Load the free local model shared by document and query embedding."""
    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        device="cpu",
        embed_batch_size=16,
        query_instruction=QUERY_INSTRUCTION,
        normalize=True,
    )


def replace_chroma_collection(nodes) -> None:
    """Embed the nodes and replace the named Chroma collection."""
    # Download and initialize the model before touching an existing collection.
    # If the first-time model download fails, previously ingested data remains.
    embed_model = create_local_embedding_model()

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Parsing happens before this function. Therefore, a bad/empty input set does
    # not erase a previously working collection.
    try:
        chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    else:
        LOGGER.info("Replacing existing Chroma collection '%s'.", COLLECTION_NAME)
        chroma_client.delete_collection(name=COLLECTION_NAME)

    chroma_collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"embedding_model": EMBEDDING_MODEL_NAME},
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Creating the index computes embeddings and stores node text, metadata, IDs,
    # and vectors in the persistent Chroma collection.
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )


def main() -> None:
    """Validate configuration, ingest PDFs, and report a useful result."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(
        path for path in DATA_DIR.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_paths:
        LOGGER.error(
            "No PDF files were found in %s. Add at least one PDF and run again.",
            DATA_DIR,
        )
        return

    nodes = load_and_chunk_pdfs(pdf_paths)
    if not nodes:
        LOGGER.error("No text chunks were created. Check that the PDFs contain extractable text.")
        return

    try:
        replace_chroma_collection(nodes)
    except Exception:
        LOGGER.exception("Ingestion failed while creating embeddings or writing ChromaDB.")
        return

    LOGGER.info(
        "Ingestion complete: stored %d chunks from %d PDF file(s) in %s",
        len(nodes),
        len(pdf_paths),
        CHROMA_DIR,
    )


if __name__ == "__main__":
    main()
