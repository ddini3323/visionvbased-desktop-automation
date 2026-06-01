# Vision-Based Desktop Automation

A Windows desktop automation system that uses a **hybrid visual grounding pipeline** to locate UI elements on screen and automate Notepad workflows. Instead of relying on fixed pixel coordinates or brittle accessibility APIs, it finds elements by description -- using computer vision and a Vision-Language Model (VLM) as fallback.

---

## How It Works

The automation fetches 10 posts from the [JSONPlaceholder API](https://jsonplaceholder.typicode.com), then for each post:

1. Minimizes all windows to show the desktop
2. Locates the **Notepad icon** using the hybrid grounding pipeline
3. Double-clicks to open Notepad
4. Pastes the post content (title + body)
5. Saves the file to `Desktop/tjm-project/post_<id>.txt`
6. Closes Notepad and repeats

### Hybrid Grounding Pipeline

Detection runs through 4 stages in order -- the first success short-circuits the rest:

| Stage | Method | Description |
|-------|--------|-------------|
| 1 | **DINO** | Zero-shot object detection via GroundingDINO (HuggingFace) |
| 2 | **OCR** | Finds the icon text label via EasyOCR, offsets upward to the icon center |
| 3 | **Template** | OpenCV multi-scale template matching against a saved icon crop |
| 4 | **VLM** | Claude (vision) -- ScreenSeekeR-inspired recursive visual search with position inference and existence verification |

After a successful detection, the icon crop is auto-saved as a template so Stage 3 gets faster on subsequent runs.

---

## Requirements

- **Windows 10 or 11** (uses `win32gui`, `pyautogui`, and DPI-aware Win32 APIs)
- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip
- An **Anthropic API key** (for the VLM fallback stage)
- A **Notepad shortcut on the desktop** (the automation double-clicks it)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ddini3323/visionvbased-desktop-automation.git
cd visionvbased-desktop-automation
```

### 2. Install dependencies

**Using uv (recommended):**

```bash
pip install uv        # install uv if you don't have it
uv sync               # creates .venv and installs all dependencies
```

**Using pip:**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

> **Note:** `torch`, `transformers` (for GroundingDINO), and `easyocr` are large downloads (~1-3 GB total). They are only loaded on first use -- the pipeline gracefully skips any stage that fails to load.

### 3. Configure environment

Create a `.env` file in the project root:

```env
# Required -- your Anthropic API key
ANTHROPIC_API_KEY=sk-ant-...

# Optional overrides (defaults shown)
VLM_MODEL=claude-sonnet-4-6
VLM_MAX_RETRIES=3
PROJECT_DIR_NAME=tjm-project
SCREENSHOT_DIR=screenshots
LOG_DIR=logs
```

> The VLM stage only fires if the earlier stages fail, so API usage is minimal in practice.

### 4. Prepare your desktop

Make sure there is a **Notepad shortcut** visible on your Windows desktop. Right-click the desktop > New > Shortcut > type `notepad.exe`.

---

## Running

```bash
# With uv
uv run python main.py

# With activated venv
python main.py
```

The script will:
- Print structured logs to the console and `logs/automation.log`
- Save annotated detection screenshots to `screenshots/post_<id>_detection.png`
- Save the 10 post files to `Desktop/tjm-project/`

### Expected output

```
============================================================
Desktop Automation - VLM Visual Grounding
============================================================
Fetching posts from JSONPlaceholder API...
Got 10 posts
--------------------------------------------------
Post 1/10  id=1  title=sunt aut facere repellat provident...
--------------------------------------------------
HybridGrounder: attempt 1/3
Icon found at (960, 540)  confidence=0.91  method=template
Notepad opened
Saved -> C:\Users\...\Desktop\tjm-project\post_1.txt
...
============================================================
DONE  success=10  failed=0  time=87.4s
============================================================
```

---

## Project Structure

```
.
├── main.py                  # Entry point -- orchestrates the full workflow
├── pyproject.toml           # Project metadata and dependencies
├── uv.lock                  # Locked dependency versions
├── .env                     # Your local config (not committed)
├── templates/               # Auto-saved icon crops for template matching
├── screenshots/             # Annotated detection screenshots per run
├── logs/                    # automation.log
└── src/
    ├── api/
    │   └── jsonplaceholder.py   # Fetches posts; falls back to built-in mock data
    ├── automation/
    │   ├── desktop.py           # Mouse, keyboard, screenshot, DPI handling
    │   └── notepad.py           # Open, type, save, close Notepad via win32gui
    ├── grounding/
    │   ├── base.py              # GroundingResult dataclass + BaseGrounder ABC
    │   ├── hybrid_grounder.py   # Cascaded pipeline: DINO -> OCR -> Template -> VLM
    │   ├── dino_grounder.py     # Stage 1: GroundingDINO zero-shot detection
    │   ├── ocr_grounder.py      # Stage 2: EasyOCR label detection
    │   ├── template_grounder.py # Stage 3: OpenCV multi-scale template matching
    │   └── vlm_grounder.py      # Stage 4: Claude vision (ScreenSeekeR algorithm)
    └── utils/
        ├── file_manager.py      # Desktop path resolution, file naming, content formatting
        └── logger.py            # Colored console + file logger setup
```

---

## Configuration Reference

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `ANTHROPIC_API_KEY` | -- | **Required.** Your Anthropic API key |
| `ANTHROPIC_BASE_URL` | _(Anthropic default)_ | Custom API base URL (e.g. proxy) |
| `VLM_MODEL` | `claude-sonnet-4-6` | Claude model used for the VLM grounding stage |
| `VLM_MAX_RETRIES` | `3` | Max retry attempts per element search |
| `PROJECT_DIR_NAME` | `tjm-project` | Subfolder name created on the Desktop |
| `SCREENSHOT_DIR` | `screenshots` | Where annotated detection images are saved |
| `LOG_DIR` | `logs` | Where `automation.log` is written |

---

## Troubleshooting

**`ANTHROPIC_API_KEY not set` error on startup**
Create a `.env` file with `ANTHROPIC_API_KEY=sk-ant-...` (see Setup step 3).

**Notepad icon not found after all 4 stages**
Ensure a Notepad shortcut is on the desktop and visible (not hidden behind other windows). The first successful run auto-saves a template, so subsequent runs are faster and more reliable.

**GroundingDINO downloads ~700 MB on first run**
This is expected. It is cached in the HuggingFace local model cache (`~/.cache/huggingface`). To skip DINO entirely, remove it from the `_stages` list in `src/grounding/hybrid_grounder.py`.

**DPI / coordinate mismatch on high-DPI screens**
`DesktopController` calls `SetProcessDPIAware()` automatically. If clicks still land in the wrong place, set your display scaling to 100% in Windows Display Settings.

**`win32gui` import error**
Run `pip install pywin32` then `python Scripts/pywin32_postinstall.py -install` from your Python scripts directory.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `anthropic` | Claude API client (VLM stage) |
| `pyautogui` | Mouse and keyboard control |
| `pywin32` | Windows API -- window detection, focus, rect |
| `mss` | Fast screenshot capture |
| `pillow` | Image manipulation and annotation |
| `opencv-python-headless` | Template matching |
| `easyocr` | OCR for text label detection |
| `transformers` + `torch` | GroundingDINO model |
| `python-dotenv` | `.env` file loading |
| `pyperclip` | Clipboard-based text input (Unicode safe) |
| `colorama` | Colored terminal output |
