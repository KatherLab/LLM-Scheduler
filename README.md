# KatherLab LLM Scheduler

A web-based tool for scheduling and running large language models (LLMs) on a shared GPU server. Built for research teams that need to share limited GPU resources across multiple models and users.

![License](https://img.shields.io/badge/license-MIT-blue)

---

## What is this?

If your lab has a powerful GPU server (e.g., an HGX node with 8 GPUs) and multiple people want to run different LLMs at different times, this tool helps you manage that.

**Think of it like a shared calendar for your GPUs.**

- **See what's running** — a visual timeline shows which models are loaded and which GPUs are in use.
- **Book a model** — pick a model from the catalog, choose a time and duration, and the scheduler handles the rest.
- **No port juggling** — users always connect to one stable address. The scheduler routes requests to the right model behind the scenes.
- **Automatic lifecycle** — models start, run for the booked duration, and stop automatically. No need to SSH in and manage processes.

---

## Who is this for?

- Research groups sharing a single GPU server
- Labs running multiple LLMs (e.g., for experiments, evaluations, demos)
- Anyone who wants a simple web UI instead of manually launching and killing vLLM processes via the terminal

---

## How it works (high level)

```
  You (browser)          KatherLab LLM Scheduler           GPU Server (Slurm)
  ┌──────────┐           ┌─────────────────────┐           ┌──────────────────┐
  │  Web UI  │──────────▶│  Scheduler + Router  │──────────▶│  vLLM instances   │
  │          │◀──────────│                     │◀──────────│  (Slurm jobs)     │
  └──────────┘           └─────────────────────┘           └──────────────────┘
```

1. **You open the web UI** and see a timeline of GPU usage and a catalog of available models.
2. **You create a booking** — e.g., "Run Qwen3.5-397B from 10:00 to 18:00."
3. **The scheduler submits a Slurm job** that starts vLLM with the right model and GPU allocation.
4. **Once the model is ready**, the scheduler routes API requests (`/v1/chat/completions`) to it.
5. **When the booking ends**, the model shuts down and the GPUs are freed for the next booking.

---

## Features

- 📅 **Visual GPU timeline** — drag-and-drop booking, resize, extend, shorten
- 🚀 **One-click model start** — pick from a catalog of pre-configured models
- ⚡ **ASAP booking** — automatically finds the earliest free slot
- 🔁 **Automatic retries** — if a model fails to start, the scheduler retries
- 🔀 **OpenAI-compatible proxy** — apps like LiteLLM, Open WebUI, or custom scripts connect to one stable endpoint
- 📋 **Live Slurm logs** — view stdout/stderr from the web UI
- 🔒 **Simple authentication** — password-protected access
- 🌙 **Dark mode** — because of course

---

## Quick start

### Prerequisites

- A Linux server with **Slurm** installed and working (`sbatch`, `squeue`, `scancel`)
- At least one GPU
- **Python 3.13+**
- [**uv**](https://docs.astral.sh/uv/) (fast Python package manager)

### 1. Install uv (if you don't have it)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository

```bash
git clone https://github.com/KatherLab/LLM-Scheduler.git
cd LLM-Scheduler
```

### 3. Set up the environment

```bash
uv sync
```

This creates a virtual environment and installs all dependencies automatically.

### 4. Configure

```bash
cp config/example.env .env
```

Open `.env` in a text editor and adjust:

| Setting | What it does |
|---|---|
| `AUTH_PASSWORD` | The password to log into the web UI |
| `PUBLIC_HOSTNAME` | The hostname or IP your users will connect to |
| `TOTAL_GPUS` | How many GPUs your server has |
| `SLURM_PARTITION` | Your Slurm partition (leave empty for default) |
| `VLLM_API_KEY` | API key that vLLM instances use (internal) |

### 5. Configure your models

Edit `config/models.yaml` to list the models you want to make available. Each entry specifies the model path, how many GPUs it needs, and any special vLLM arguments:

```yaml
models:
  - name: My-Model
    model_path: /path/to/model
    gpus: 2
    tensor_parallel_size: 2
    cpus: 16
    mem: "64G"
    venv_activate: /path/to/vllm/.venv/bin/activate
    notes: "Short description for the UI"
```

### 6. Run

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 9000
```

Then open **http://your-server:9000** in your browser.

---

## Connecting your apps

Once a model is running, you can send requests to the scheduler just like you would to OpenAI:

```bash
curl http://your-server:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "My-Model",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

This works with any OpenAI-compatible client, including:
- [LiteLLM](https://github.com/BerriAI/litellm)
- [Open WebUI](https://github.com/open-webui/open-webui)
- Python's `openai` library

Just point `api_base` to `http://your-server:9000` and set the `model` to one of your catalog names.

---

## Project structure

```
├── app/                  # Python backend (FastAPI)
│   ├── main.py           # App entry point + background workers
│   ├── admin.py          # Booking/lease management API
│   ├── slurm.py          # Slurm integration (sbatch, scancel, etc.)
│   ├── planner.py        # GPU allocation algorithm
│   ├── proxy.py          # OpenAI-compatible request proxy
│   └── ui/               # Web frontend (HTML + JS)
├── config/
│   ├── models.yaml       # Model catalog
│   └── example.env       # Environment variable template
├── templates/
│   └── vllm_job.sh       # Slurm job script template
└── pyproject.toml        # Python dependencies
```

---

## Contributing

Contributions are welcome! Please open an issue or pull request.

---

## License

MIT — see [LICENSE](LICENSE) for details.