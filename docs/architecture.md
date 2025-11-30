## Support AI Agent Architecture

### High-Level Flow
```mermaid
flowchart LR
    A[Support Rep / Customer] -->|Question| B[Streamlit UI]
    B -->|REST call| C[LangChain Pipeline]
    C --> D[Intent + Prompt Template]
    D --> E[ChatOpenAI GPT-4o-mini]
    C --> F[Chroma Retriever]
    F -->|Top-k docs| D
    E -->|Answer + citations| B
    E -->|Confidence score| G{Confidence < Threshold?}
    G -->|Yes| H[Escalation Hook (email/webhook)]
    G -->|No| I[Log + close]
    B --> I
```

### Components
1. **Data Layer**
   - `data/support_kb.csv` holds curated FAQs/ticket summaries.
   - `src/ingest.py` chunkifies and embeds entries via `OpenAIEmbeddings`, storing vectors in Chroma (`./vectorstore`).

2. **Reasoning Layer**
   - LangChain RetrievalQA with `ChatOpenAI (gpt-4o-mini)` and a guard-rail prompt to stay within provided context.
   - Similarity scores (`similarity_search_with_score`) converted into a 0–1 confidence metric.

3. **Support Logic**
   - Responses below `CONFIDENCE_THRESHOLD` trigger `trigger_escalation`, capturing payloads for downstream notification systems.
   - Metadata (doc id, tags, priority) attached for transparency and triage.

4. **Interface**
   - Streamlit chat UI showing history, confidence labels, and source tags.
   - Sidebar tracks pending escalations for live operators.

5. **Observability & Future Hooks**
   - Placeholder for logging to Sheets/Supabase.
   - Add tracing (LangSmith/OpenTelemetry) if needed for debugging prompt performance.

Export this diagram (e.g., via [https://mermaid.live](https://mermaid.live)) for final submission assets. Update the file as the architecture evolves.

