# Garmin FIT to CSV

Streamlit web app that converts Garmin `.fit` files into a clean running-focused CSV output.

https://fit2csv.streamlit.app/

## Local setup

```bash
python -m venv .venv
```

Activate virtual environment:
- PowerShell: `./.venv/Scripts/Activate.ps1`
- cmd: `.venv\\Scripts\\activate.bat`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run locally:
```bash
streamlit run app.py
```

## Notes

- App accepts only running FIT files (`sport == running`)
- Output starts with core running columns, then appends all other decoded record columns
- Unknown FIT fields (`unknown_*`) are intentionally excluded
- Created mainly for personal use cases
