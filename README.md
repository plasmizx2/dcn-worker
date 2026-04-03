# DCN Worker

Distributed worker node for [DCN](https://github.com/plasmizx2/dcn-demo). This repository contains **only** what you need to run a worker — no web app, no database.

## Quick start

```bash
git clone https://github.com/plasmizx2/dcn-worker.git
cd dcn-worker
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp config.json.example config.json
# Edit config.json: set server_url to your DCN API (e.g. https://dcn-demo.onrender.com) and worker_name
python run.py
```

**GUI (browser):** `python app.py` — opens http://127.0.0.1:7777. The dashboard polls server state every second (no stale cache). Use **Stop Worker** before exiting so status, uptime, and the activity log stay aligned with the engine.

## Config

Copy `config.json.example` to `config.json` and set:

| Field | Description |
|-------|-------------|
| `server_url` | Base URL of the DCN API (no trailing slash) |
| `worker_name` | Display name for this machine in the operator dashboard |

`config.json` is gitignored so your settings stay local.

## Requirements

- Python 3.10+
- Network access to the DCN server
- Dependencies listed in `requirements.txt` (scikit-learn, pandas, OpenML, httpx for ML / remote CSV jobs, etc.)

## Sync with the main project

Handler and dataset logic is maintained in parallel with [dcn-demo](https://github.com/plasmizx2/dcn-demo). When the server’s ML pipeline changes, update `datasets.py` and `handlers/` here to match.

## License

Same as the parent DCN project.
