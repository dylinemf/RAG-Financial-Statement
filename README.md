# RAG-Financial-Statement — Additional Notes

This project is based on [InterOpera-Apps/coding-test-2nd](https://github.com/InterOpera-Apps/coding-test-2nd).

**For setup, usage, and general documentation, please refer to the [original repository](https://github.com/InterOpera-Apps/coding-test-2nd).**  
Below are the additional features and differences in this repository:

- **Project Structure**
```
RAG-Financial-Statement/
├── backend/
│ ├── main.py # FastAPI application
│ ├── models/ # Data models
│ │ └── schemas.py # Pydantic schemas
│ ├── services/ # RAG service logic
│ │ ├── pdf_processor.py # PDF processing and chunking
│ │ ├── providers.py # Embeddings and LLM management (HF & OpenAI)
│ │ ├── vector_store.py # Vector database integration
│ │ └── rag_pipeline.py # RAG pipeline
│ ├── requirements.txt # Python dependencies
│ └── config.py # Configuration file
├── frontend/
│ ├── pages/ # Next.js pages
│ │ ├── index.tsx # Main page
│ │ └── _app.tsx # App component
│ ├── components/ # React components
│ │ ├── ChatInterface.tsx
│ │ └── FileUpload.tsx
│ ├── hooks/ # Hooks
│ │ └── useKnowledgeBase.ts # Trigger based on batch process
│ ├── styles/ # CSS files
│ │ └── globals.css # Global styles
│ │ └── ChatInterface.module.css # Module style
│ │ └── FileUpload.module.css # Module style
│ ├── public/ # Hooks
│ │ └── favicon.png
│ ├── package.json # Node.js dependencies
│ ├── tsconfig.json # TypeScript configuration
│ ├── next.config.js # Next.js configuration
│ ├── next-env.d.ts # Next.js type definitions
│ └── .eslintrc.json # ESLint configuration
├── data/
│ └── sample.pdf
└── README.md
```

## 🚩 What's Different in This Repo

- **Modular Provider Management (`providers.py`):**
  - Added a `providers.py` file to make embedding and LLM provider (HuggingFace or OpenAI) selection more modular and extensible. You just configure via env variables.

- **Not Using `_retrieve_documents`:**
  - The function `_retrieve_documents` from the original repo is not used here.
  - All retrieval is handled in `vector_store.py` via `similarity_search`, called directly from the RAG pipeline (`generate_answer`). So there's no dependency on that function.

- **Frontend Response Formatting:**
  - Added `react-markdown`, `remark-math`, and `katex` to render model responses with markdown and math formatting, making answers much cleaner and easier to read.

- **Embeddings & LLM Provider Choice:**
  - You can select either HuggingFace or OpenAI for both embeddings and LLM, simply by changing values in the `.env` file (see `.env.sample`). To use OpenAI, supply your own `OPENAI_API_KEY`.

- **Better Prompt Engineering:**
  - In `_generate_llm_response` (see `rag_pipeline.py`), the prompt is improved: if an answer can be calculated manually from data in the context, the model is instructed to do the math and show the steps (with LaTeX/markdown), always showing the final result in bold at the top.

- **Knowledge Base State Hook:**
  - Added a `useKnowledgeBase` React hook in the frontend to handle state changes when PDF processing/batching is done, so the chat interface is only enabled when knowledge base is ready.

- **Component-level Styling:**
  - Styling for each React component (chat, file upload, etc) is implemented as a CSS module for better maintainability.

---

> Again: for overall instructions, setup, and original API routes, see the [main repo documentation](https://github.com/InterOpera-Apps/coding-test-2nd).