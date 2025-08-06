import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore import InMemoryDocstore
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import faiss

# ==============================
# CONFIG
# ==============================
LLM_MODEL_PATH = "./models/local-llm"     # Your local Hugging Face LLM (e.g., LLaMA, Mistral)
EMBED_MODEL_PATH = "./models/distilbert-base-nli-mean-tokens"  # DistilBERT NLI embedder
CODEBASE_PATH = "./my_codebase"
INDEX_PATH = "./code_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
BATCH_SIZE = 16

# ==============================
# HELPER: NLI Embedder (Mean Pooling)
# ==============================
class NLIEmbedder:
    def __init__(self, model_path):
        print(f"Loading embedding model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

    def embed(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['
