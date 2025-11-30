import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

load_dotenv()

st.title("My AI Agent")
st.write("Demo for my AI Agent Development Challenge")
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "vectorstore"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
LOG_PATH = Path(os.getenv("CHAT_LOG_PATH", "data/chat_logs.csv"))
ESCALATION_WEBHOOK_URL = os.getenv("ESCALATION_WEBHOOK_URL")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").lower()
LOCAL_LLM_NAME = os.getenv("LOCAL_LLM_NAME", "google/flan-t5-base")
RECOMMENDED_QUESTIONS: List[str] = [
    "How do I reset my portal password?",
    "What is the SLA for premium customers?",
    "Can I upgrade my plan mid-cycle?",
    "How do I enable two-factor authentication?",
    "Where do I download invoices?",
    "Can I connect my Slack workspace?",
]

SYSTEM_PROMPT = """You are SupportGPT, a helpful support specialist.
Use ONLY the context provided. Cite the question IDs when relevant.
If the answer is not present, say you do not know and suggest escalation."""

QA_PROMPT = PromptTemplate.from_template(
    "{system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
)


def build_vectorstore_if_missing():
    """Auto-build vectorstore if it doesn't exist."""
    if VECTORSTORE_DIR.exists():
        return
    
    import sys
    import subprocess
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    ingest_script = project_root / "src" / "ingest.py"
    
    if not ingest_script.exists():
        st.error("Ingestion script not found. Cannot auto-build vectorstore.")
        return
    
    with st.spinner("Building knowledge base from CSV... This may take 2-3 minutes."):
        try:
            result = subprocess.run(
                [sys.executable, str(ingest_script)],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                st.error(f"Failed to build vectorstore: {result.stderr}")
            else:
                st.success("Knowledge base built successfully!")
        except Exception as e:
            st.error(f"Error building vectorstore: {e}")


def load_vectorstore() -> Chroma:
    build_vectorstore_if_missing()
    
    if not VECTORSTORE_DIR.exists():
        raise RuntimeError(
            "Vector store not found and auto-build failed. Check logs."
        )
    
    # Use HuggingFace embeddings (local, no API needed)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )


@st.cache_resource(show_spinner=False)
def load_local_pipeline(model_name: str = LOCAL_LLM_NAME):
    """Load local HuggingFace model pipeline."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    gen_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
    )
    return HuggingFacePipeline(pipeline=gen_pipeline)


def get_llm():
    """Get LLM - uses local HuggingFace by default."""
    if LLM_PROVIDER == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY missing. Set it or use LLM_PROVIDER=local."
            )
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    if LLM_PROVIDER == "local":
        st.info(f"🤖 Using local HuggingFace model: {LOCAL_LLM_NAME}")
        return load_local_pipeline(LOCAL_LLM_NAME)
    
    raise RuntimeError(
        f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Use 'openai' or 'local'."
    )


def build_chain(vectorstore: Chroma) -> RetrievalQA:
    llm = get_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": QA_PROMPT.partial(system_prompt=SYSTEM_PROMPT)},
        return_source_documents=True,
    )


def compute_confidence(results: List[Tuple[Any, float]]) -> float:
    if not results:
        return 0.0
    avg_distance = sum(score for _, score in results) / len(results)
    return max(0.0, 1 - min(avg_distance, 1))


def trigger_escalation(payload: Dict[str, Any]) -> None:
    payload.setdefault("timestamp", datetime.utcnow().isoformat())
    payload.setdefault("status", "pending")
    st.session_state.escalations.append(payload)
    if ESCALATION_WEBHOOK_URL:
        try:
            requests.post(
                ESCALATION_WEBHOOK_URL,
                json=payload,
                timeout=5,
            )
        except requests.RequestException as exc:
            st.warning(f"Escalation webhook failed: {exc}")


def append_log(entry: Dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_PATH.exists()
    fieldnames = [
        "timestamp",
        "question",
        "answer",
        "confidence",
        "sources",
        "escalated",
    ]
    with LOG_PATH.open("a", encoding="utf-8", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)


def main() -> None:
    st.set_page_config(page_title="Support AI Agent", page_icon="🤖")
    st.title("Support AI Agent")
    st.caption("RAG-powered assistant for customer support teams.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "escalations" not in st.session_state:
        st.session_state.escalations = []

    try:
        vectorstore = load_vectorstore()
    except Exception as exc:  # pragma: no cover - streamlit feedback
        st.error(str(exc))
        st.stop()

    qa_chain = build_chain(vectorstore)

    with st.sidebar:
        st.subheader("Escalations")
        if st.session_state.escalations:
            df = pd.DataFrame(st.session_state.escalations)
            st.dataframe(
                df[["timestamp", "question", "reason", "confidence", "status"]],
                use_container_width=True,
            )
            for idx, esc in enumerate(st.session_state.escalations):
                if esc["status"] != "resolved":
                    if st.button(
                        f"Resolve #{idx+1}",
                        key=f"resolve_{idx}",
                    ):
                        st.session_state.escalations[idx]["status"] = "resolved"
                        st.rerun()
        else:
            st.write("No pending escalations.")

    st.subheader("Quick suggestions")
    suggestion_cols = st.columns(3)
    suggested_question = None
    for idx, question in enumerate(RECOMMENDED_QUESTIONS):
        if suggestion_cols[idx % 3].button(
            question, key=f"suggest_{idx}"
        ):
            suggested_question = question

    user_input = suggested_question or st.chat_input("Ask a customer support question...")
    if user_input:
        docs_with_score = vectorstore.similarity_search_with_score(user_input, k=3)
        response = qa_chain({"query": user_input})
        confidence = compute_confidence(docs_with_score)
        sources = [
            f"{doc.metadata.get('doc_id')} | {doc.metadata.get('tags')}"
            for doc, _ in docs_with_score
        ]
        answer = response["result"]
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input, "confidence": 1.0}
        )
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "confidence": confidence,
                "sources": sources,
                "question": user_input,
            }
        )
        escalated = confidence < CONFIDENCE_THRESHOLD
        if escalated:
            trigger_escalation(
                {
                    "question": user_input,
                    "response": answer,
                    "confidence": confidence,
                    "reason": "Low retrieval confidence",
                }
            )
        append_log(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "question": user_input,
                "answer": answer,
                "confidence": round(confidence, 4),
                "sources": ";".join(sources),
                "escalated": escalated,
            }
        )

    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant":
                st.caption(f"Confidence: {msg['confidence']:.2f}")
                if msg.get("sources"):
                    st.caption("Sources: " + ", ".join(msg["sources"]))
                st.caption("Manual escalation available if more help is needed.")
                if st.button(
                    "Escalate to human",
                    key=f"manual_escalate_{idx}",
                ):
                    trigger_escalation(
                        {
                            "question": msg.get("question", "N/A"),
                            "response": msg["content"],
                            "confidence": msg.get("confidence", 0.0),
                            "reason": "Manual escalation",
                        }
                    )
                    st.rerun()


if __name__ == "__main__":
    main()

