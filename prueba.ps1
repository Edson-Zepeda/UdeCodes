#!/usr/bin/env pwsh

$ErrorActionPreference = 'Stop'

if (-not (Test-Path '.\.venv\Scripts\python.exe')) {
    Write-Error 'Virtualenv not found (.\\.venv). Run .\\.venv\\Scripts\\Activate.ps1 first.'
    exit 1
}

$python = Join-Path (Resolve-Path '.\\.venv\Scripts') 'python.exe'

if (-not $env:GENAI_API_KEY) {
    Write-Warning 'GENAI_API_KEY is not set. Use $env:GENAI_API_KEY="YOUR_KEY" before running to call Gemini.'
}

& $python scripts/run_simulation_summary.py
