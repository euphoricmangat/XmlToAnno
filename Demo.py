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
MODEL_PATH = "./models/llama"           # Local Hugging Face model for Q&A
BERT_PATH = "./models/bert-base-uncased" # Local BERT model for embeddings
CODEBASE_PATH = "./my_codebase"          # Path to your codebase
INDEX_PATH = "./code_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

# ==============================
# HELPER: BERT Embedder
# ==============================
class BertEmbedder:
    def __init__(self, model_path):
        print(f"Loading BERT model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

    def embed(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

# ==============================
# STEP 1: Load Q&A Model (CPU)
# ==============================
print("Loading main language model on CPU...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype="auto", device_map="cpu")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, temperature=0.2, do_sample=True)
llm = HuggingFacePipeline(pipeline=pipe)

# ==============================
# STEP 2: Index Codebase (or load existing)
# ==============================
if not os.path.exists(INDEX_PATH):
    print("Indexing codebase...")
    docs = []
    for root, dirs, files in os.walk(CODEBASE_PATH):
        for file in files:
            if file.endswith((".py", ".java", ".js", ".ts", ".html", ".css", ".md")):
                with open(os.path.join(root, file), "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    docs.append(Document(page_content=content, metadata={"source": file}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    print(f"Total chunks created: {len(chunks)}")

    embedder = BertEmbedder(BERT_PATH)
    texts = [chunk.page_content for chunk in chunks]
    embeddings = normalize(embedder.embed(texts))

    db = FAISS.from_embeddings(texts, embeddings)
    db.save_local(INDEX_PATH)
else:
    print("Loading existing index...")
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
