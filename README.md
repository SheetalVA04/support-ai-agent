## Support AI Agent

AI-powered Support Assistant that answers customer FAQs from an internal knowledge base, explains the context it used, and flags low-confidence conversations for human escalation. Built for the AI Agent Development Challenge (Sales, Marketing & Support track).

### Key Features
- Retrieval-augmented responses grounded in curated FAQ/product docs
- Confidence scoring with automatic escalation prompts when context is weak
- Conversation logging with metadata for later analytics
- Streamlit chat UI with quick-question suggestions, manual escalation buttons, and cited snippets for transparency

### Architecture
1. **Ingestion**: `data/support_kb.csv` → `src/ingest.py` → embeddings stored in `Chroma` (`./vectorstore`)
2. **Reasoning**: LangChain pipeline using `ChatOpenAI` (`gpt-4o-mini` by default) with a guard-rail prompt
3. **Support Logic**: similarity-based confidence score controls escalation guidance + optional webhook/email notification hook
4. **Interface**: Streamlit chat app (`app/app.py`) with session history and source traceability

See `docs/architecture.md` for the detailed flow diagram.

### Tech Stack
- **LLM & Embeddings**: OpenAI GPT-4o-mini, `text-embedding-3-small`
- **Offline Mode**: HuggingFace `google/flan-t5-base` via Transformers + local sentence-transformer embeddings
- **Orchestration**: LangChain + LangChain OpenAI bindings
- **Vector Store**: Chroma (local persistence)
- **UI**: Streamlit
- **Persistence**: CSV log + optional integrations (Sheets/Supabase hooks placeholder)

### Setup
1. Clone this repo and install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the project root and fill in:
   - `OPENAI_API_KEY`
   - `SUPPORT_DATA_PATH` (defaults to `data/support_kb.csv`)
   - `VECTORSTORE_DIR` (defaults to `vectorstore`)
   - `CONFIDENCE_THRESHOLD` (optional, default `0.55`)
   - `CHAT_LOG_PATH` (optional, default `data/chat_logs.csv`)
   - `ESCALATION_WEBHOOK_URL` (optional, post JSON payload on low confidence)
   - `LLM_PROVIDER` (`openai` or `local`, default `openai`)
   - `LOCAL_LLM_NAME` (defaults to `google/flan-t5-base` when `LLM_PROVIDER=local`)
3. Populate `data/support_kb.csv` with your own FAQs or ticket summaries.
4. Build the vector store:
   ```
   python src/ingest.py
   ```
5. Launch the UI:
   ```
   streamlit run app/app.py
   ```

### Running Locally
- `streamlit run app/app.py` opens the chat UI at `http://localhost:8501`.
- Queries retrieve relevant snippets and display source tags (question IDs / priority).
- When confidence falls below the threshold (default 0.55), the UI suggests escalating; hook in your alerting workflow inside `trigger_escalation`.
- Every exchange is appended to `CHAT_LOG_PATH` (default `data/chat_logs.csv`) for analytics or auditing.
- Set `LLM_PROVIDER=local` to run entirely offline (downloads the `LOCAL_LLM_NAME` model the first time, ~1GB for `flan-t5-base`).
- Use the “Quick suggestions” buttons to instantly populate common questions.
- Manual “Escalate to human” buttons appear with each bot reply, and the sidebar shows an escalation audit dashboard with resolve controls.

### Repository Deliverables
- `app/app.py` – Streamlit interface + reasoning logic
- `src/ingest.py` – Data ingestion + embedding script
- `data/` – Sample knowledge base CSV (replace with real data)
- `docs/architecture.md` – Diagram + explanation
- `README.md` – Overview, setup, limitations, and future work (update as project evolves)

### Future Improvements
- Integrate ticketing APIs (Freshdesk/Zendesk) for live context
- Multi-lingual responses via translation layer
- Reinforcement loop that captures agent feedback and retrains the knowledge base
- Voice/WhatsApp channels via Twilio + speech-to-text

### Challenge Checklist
- [ ] Working Streamlit demo (hosted or local video)
- [ ] Public/private Git repo with instructions
- [ ] Architecture diagram exported from `docs/architecture.md`
- [ ] README completed with limitations + improvement ideas
- [ ] (Optional) 2–3 min walkthrough recording

