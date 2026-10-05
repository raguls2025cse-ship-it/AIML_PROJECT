Set-Location "$PSScriptRoot"
if (!(Test-Path ".venv")) { python -m venv .venv }
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"
Set-Location "$PSScriptRoot\backend"
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
