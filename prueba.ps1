#!/usr/bin/env pwsh
\n# Ejecuta la simulacion integrada (Gemini + SPIR) y muestra los datos clave.
if (-not (Test-Path '.\\.venv\\Scripts\\python.exe')) {
    Write-Error 'No se encontro la venv. Ejecuta primero .\\.venv\\Scripts\\Activate.ps1'
    exit 1
}

 = '.\\.venv\\Scripts\\python.exe'
&  scripts/run_simulation_summary.py
