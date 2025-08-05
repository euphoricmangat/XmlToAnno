import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline

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
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings.cpu().numpy()

def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def embed_in_batches(embedder, texts, batch_size):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = embedder.embed(batch)
        all_embeddings.append(batch_embeddings)
    return np.vstack(all_embeddings)

# ==============================
# STEP 1: Load Local LLM (CPU)
# ==============================
print("Loading local Hugging Face model on CPU...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, torch_dtype="auto", device_map="cpu")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, temperature=0.2, do_sample=True)
llm = HuggingFacePipeline(pipeline=pipe)

# ==============================
# STEP 2: Build or Load FAISS Index Incrementally
# ==============================
if not os.path.exists(INDEX_PATH):
    print("Creating new FAISS index...")
    embedder = NLIEmbedder(EMBED_MODEL_PATH)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    db = None
    for root, dirs, files in os.walk(CODEBASE_PATH):
        for file in files:
            if file.endswith((".py", ".java", ".js", ".ts", ".html", ".css", ".md")):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                chunks = text_splitter.split_text(content)
                if not chunks:
                    continue

                embeddings = embed_in_batches(embedder, chunks, BATCH_SIZE)
                embeddings = normalize(embeddings)

                if db is None:
                    db = FAISS.from_embeddings(chunks, embeddings)
                else:
                    db.add_texts(chunks, embeddings)

    db.save_local(INDEX_PATH)
else:
    print("Loading existing FAISS index...")
    db = FAISS.load_local(INDEX_PATH, embeddings=None, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_kwargs={"k": TOP_K})

# ==============================
# STEP 3: Create QA Chain
# ==============================
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ==============================
# STEP 4: Interactive Q&A
# ==============================
print("\n✅ Ready! Ask questions about your codebase (type 'exit' to quit):")
while True:
    query = input("\n> ")
    if query.lower() in ["exit", "quit"]:
        break
    answer = qa.run(query)
    print(f"\nAnswer:\n{answer}\n")
