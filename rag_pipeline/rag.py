# rag_pipeline/rag.py

from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .vector_store import load_vectorstore


def get_llm_pipeline():
    model_name = "microsoft/Phi-3-mini-4k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.2,
    )

    return HuggingFacePipeline(pipeline=pipe)


def format_sources(docs):
    metadata_list = []
    for d in docs:
        meta = d.metadata
        entry = {
            "id": meta.get("id", ""),
            "country": meta.get("country", ""),
            "year": meta.get("year", ""),
            "snippet": meta.get("snippet", d.page_content),
        }
        metadata_list.append(entry)
    return metadata_list


def answer_question(question: str):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join(d.page_content for d in docs)

    llm = get_llm_pipeline()

    system_prompt = (
        "You are a climate expert. Answer ONLY using the context below. "
        "If the answer is not in the context, say you do not know.\n\n"
        f"Context:\n{context}\n\n"
    )

    response = llm(system_prompt + question)

    return response[0]["generated_text"], format_sources(docs)

# --- API wiring (create pipeline once, reuse) ---
from .embeddings import get_embedding_model
from .vector_store import FaissVectorStore
from .retriever import SimpleRetriever

_PIPELINE: RAGPipeline | None = None

def get_pipeline() -> RAGPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        emb = get_embedding_model()
        vs = FaissVectorStore.load_local(embedding=emb)  # loads from vector_store/faiss_index
        retriever = SimpleRetriever(vector_store=vs)
        _PIPELINE = RAGPipeline(retriever=retriever, llm=dummy_llm)  # swap llm later if needed
    return _PIPELINE

def answer_question(question: str, k: int = 3) -> Dict[str, Any]:
    return get_pipeline().ask(question, k=k)

