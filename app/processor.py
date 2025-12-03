import os
import base64
import uuid
import logging
from typing import List, Dict, Any
from .layout import extract_pdf_blocks
from .llm_client import OpenRouterClient
from .vector_store import upsert_node, ensure_collection
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBED_MODEL_NAME = "all-mpnet-base-v2"

class Processor:
    def __init__(self, openrouter_client: OpenRouterClient):
        self.orm = openrouter_client
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        # ensure qdrant collection is created with default dim (will be reset by Cursor if needed)
        ensure_collection(dim=self.embedder.get_sentence_embedding_dimension())

    def process_pdf(self, doc_id: str, path: str, model_id: str):
        blocks = extract_pdf_blocks(path)
        nodes = []
        for b in blocks:
            if b["type"] == "text":
                content = b["content"]
                emb = self.embed_text(content)
                payload = {
                    "doc_id": doc_id,
                    "block_type": "text",
                    "content_text": content
                }
                nid = upsert_node(emb.tolist() if hasattr(emb, "tolist") else emb, payload)
                nodes.append({"node_id": nid, "type":"text"})
            else:
                # image block: call vision LLM -> expect strict JSON
                image_b64 = base64.b64encode(b["content_bytes"]).decode()
                prompt = "Analyze this image. If flowchart or table, return strict JSON following SPEC's schema. Otherwise say it's an image without diagram. Only JSON."
                try:
                    resp = self.orm.analyze_image_strict_json(model_id, image_b64, prompt)
                except Exception as e:
                    logger.exception("Vision call failed")
                    continue
                # Cursor should map resp to actual json (resp parsing may be needed)
                # For now, attempt to extract content_text summary and structured_json
                # This is a TODO for Cursor to refine parsing and validation
                structured_json = resp  # placeholder
                summary = "AUTOGEN_SUMMARY"  # Cursor should replace
                emb = self.embed_text(summary)
                payload = {
                    "doc_id": doc_id,
                    "block_type": "diagram",
                    "content_text": summary,
                    "structured_json": structured_json
                }
                nid = upsert_node(emb.tolist() if hasattr(emb, "tolist") else emb, payload)
                nodes.append({"node_id": nid, "type":"diagram"})
        return nodes

    def embed_text(self, text: str):
        vect = self.embedder.encode(text)
        return vect
