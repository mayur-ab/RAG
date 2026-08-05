Next steps to run the RAG
You're on Step 1 (ingest). Here's what comes next:

Step 2 — Run the benchmark (quick CLI test)
With your venv still active:

python scripts/run_benchmark.py
This runs 3 sample questions against the ingested document and prints answers plus evaluation metrics (precision, recall, faithfulness, etc.).

Step 3 — Start the REST API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
Then open http://localhost:8000/docs for the Swagger UI.

Step 4 — Query via API
Example with curl or PowerShell:

Invoke-RestMethod -Method POST -Uri "http://localhost:8000/query" -ContentType "application/json" -Body '{"query": "What are the four mental muscles in the Rishi Cognitive Framework?"}'
Or use the /query endpoint in Swagger.