# Contributor guide

## Project context
Here is a short 2-3 days hackaton project to predict patient ignorance of mammography

Key modules:
- TBD


** Do not ever read or index `data` folder and `.env` file!

## Contributor behavior
- Always try to understand the entire code structure and adhere to existing code structure, use .env, config.yaml when nessecary
- Before answering questions, always read the relevant file first rather than asking the user to share the code
- Don't restore code that you wrote and I deleted
- Don't delete my comments
- Don't make commits or git changes unless explicitly asked
- Don't hide errors with try-catch blocks; let failures surface until asked otherwise
- Don't over-engineer: no extra abstractions, features, or "improvements" beyond what's requested
- Don't add comments, docstrings, or type annotations to code you didn't change
- Ask clarifying questions when requirements are ambiguous instead of guessing

## ⚠️ CRITICAL: Always use `uv run` for Python commands

**NEVER run Python directly** (`python`, `python3`, `python -m`). **ALWAYS use `uv run python`** instead.

❌ **WRONG:** `python script.py`, `python3 -m pytest`, `python -c "import foo"`
✅ **CORRECT:** `uv run python script.py`, `uv run pytest`, `uv run python -c "import foo"`

This applies to:
- Running Python scripts
- Running tests with pytest
- Syntax checks with `python -m py_compile`
- Importing modules
- Any Python execution whatsoever

**Why:** This project uses uv for dependency management. Running Python directly bypasses the virtual environment and will cause import errors or use wrong package versions.


## Testing
- Don't add new test files unless explicitly asked
- Never import inside test functions; add imports at the top of the file



