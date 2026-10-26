# Research Agent

Local-first, agentic research over an indexed O'Reilly AI book collection.

## Streamlit app

The Streamlit frontend is in [`app.py`](./app.py). 

Its features are

- **Fast** and **Deep** research profiles from `research_agent.modes`
- A persistent chat thread for follow-up questions
- Bounded conversation context: the latest four completed turns are supplied to
  the agent to resolve references and guide direction
- Local hybrid retrieval with BM25 + vector search
- Optional Tavily web research for every question
- Live graph-step progress while the agent runs
- A cited synthesis and an evidence ledger with book page or web provenance

### Run locally

1. Install the project and development dependencies:

   ```powershell
   uv sync --extra dev
   ```

2. Set the required model in `.env` (copy `.env.example` first if needed):

   ```text
   OLLAMA_MODEL=qwen3.5:4b
   OLLAMA_HOST=http://localhost:11434
   ```

   I will be adding another implemetation with llama.cpp

   Ollama must be running and the local retrieval index must already exist in
   `data/index/`.

3. Launch the interface:

   ```powershell
   uv run streamlit run app.py
   ```

If web research is enabled in the sidebar, set `TAVILY_API_KEY` as well.
Leave it disabled for a fully local run.

The conversation is stored in the current Streamlit session only. Use
**Clear conversation** in the sidebar to start a fresh research thread.

### Research cycles

Each question runs through the selected research profile. Both profiles search
the local index first, score the retrieved passages for relevance, and then
combine the best local evidence with Tavily results when web research is
enabled. The final response is synthesized from the combined evidence and
includes citations for the passages used.

#### Fast

Fast mode is intended for quick answers:

1. Use the question as the retrieval query without generating variants.
2. Retrieve matching passages from the local BM25 and vector index.
3. Grade the local passages for relevance.
4. Search Tavily once when web research is enabled, including for questions
   unrelated to the indexed books.
5. Generate the cited answer from the selected local and web evidence.

Fast mode uses up to 6 final evidence passages, up to 4 Tavily results, and
allows one rewrite attempt when web research is disabled and local evidence is
not sufficient.

#### Deep

Deep mode spends more model calls improving recall and query coverage:

1. Expand the question into three retrieval-oriented query variants.
2. Retrieve passages for all variants from the local BM25 and vector index.
3. Grade the combined local passages for relevance.
4. Search Tavily once when web research is enabled, including for questions
   unrelated to the indexed books.
5. If web research is disabled and local evidence remains insufficient, retry
   retrieval with up to three strategies: step-back, specific, and fusion.
6. Generate the cited answer from up to 10 selected evidence passages.

Deep mode can return more comprehensive results, but it uses additional
retrieval and query-expansion calls and may take longer than Fast mode.
