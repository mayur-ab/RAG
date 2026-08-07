# Build a local Ollama model with num_ctx=32768 baked in.
# Usage: .\scripts\create_ollama_rag_model.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Modelfile = Join-Path $Root "ollama\Modelfile.rag"

Write-Host "Creating Ollama model 'rag-llama' from $Modelfile ..."
ollama create rag-llama -f $Modelfile
Write-Host "Done. Set OLLAMA_LLM_MODEL=rag-llama in your .env to use it."
