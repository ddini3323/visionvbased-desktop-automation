# Project Overview — Vision-Based Desktop Automation

## What It Does

This project automates a repetitive desktop task end-to-end: **fetching data from the internet and saving it to files using Notepad** — without any human clicking.

It fetches 10 blog posts from a public API, then for each post: opens Notepad, types the content, saves it as a `.txt` file on the Desktop, and closes Notepad. All 10 files end up in `Desktop/tjm-project/` automatically.

---

## The Core Innovation — Vision-Based Element Detection

Most desktop automation tools use fixed pixel coordinates ("click at x=450, y=300") or UI accessibility APIs. This project does something smarter: it **finds UI elements by natural language description** using a 4-stage hybrid vision pipeline.

### The 4-Stage Pipeline

| Stage | Method | How it works |
|-------|--------|-------------|
| 1 | **GroundingDINO** | AI object detection model — understands "Notepad icon" as a visual concept, zero-shot |
| 2 | **OCR** | Reads text labels on screen via EasyOCR, locates the icon by its label text |
| 3 | **Template Matching** | OpenCV multi-scale matching — remembers what the icon looked like last time |
| 4 | **Claude VLM** | Takes a screenshot, sends it to Claude, asks "where is the Notepad icon?" — Claude looks at the image and points to it |

Each stage is tried in order. The **first success short-circuits the rest**. If DINO finds the icon instantly, Claude is never called (saving API cost and time). After any successful detection, the icon crop is auto-saved as a template so Stage 3 gets a head start on the next run.

---

## Why This Approach Matters

### Traditional automation breaks easily
- Hardcoded pixel coordinates fail when resolution changes, window moves, or theme updates
- Accessibility APIs only work when apps expose them correctly
- Image templates fail when icons change appearance

### This project is resilient
- **No hardcoded coordinates** — works on any Windows machine without reconfiguring
- **Adaptive** — if one detection method fails, the next takes over automatically
- **Self-improving** — saves a template after each successful detection so future runs are faster
- **Graceful degradation** — each stage loads independently; if DINO fails to load, OCR runs instead

---

## Technical Architecture

```
main.py  (orchestrator)
│
├── src/api/          JSONPlaceholder API client with retry + mock fallback
├── src/automation/   Mouse, keyboard, screenshot, Notepad window control (win32gui)
├── src/grounding/    4-stage hybrid vision pipeline
│   ├── DINO          HuggingFace GroundingDINO — zero-shot object detection
│   ├── OCR           EasyOCR text detection + icon offset calculation
│   ├── Template      OpenCV multi-scale template matching
│   └── VLM           Claude vision — ScreenSeekeR recursive visual search
└── src/utils/        File manager (OneDrive-aware Desktop path), logger
```

The VLM stage is based on the **ScreenSeekeR algorithm** from the research paper *"ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use"* (arXiv:2504.07981). It works by:
1. Sending the screenshot + a position inference prompt to Claude
2. Claude identifies candidate regions using XML tags (`<element>`, `<area>`, `<neighbor>`)
3. Each candidate is scored with a Gaussian kernel and ranked
4. The top region is cropped and sent back to Claude for existence verification
5. If confirmed, the center coordinate is returned; otherwise the next region is tried

---

## What It Demonstrates

This is a **proof-of-concept for vision-based RPA (Robotic Process Automation)** — the kind of technology enterprise automation platforms charge significantly for, built from scratch using:

- Open-source computer vision models (GroundingDINO, EasyOCR, OpenCV)
- A Vision-Language Model (Claude) as an intelligent fallback
- Standard Python automation libraries (pyautogui, win32gui)

The key insight: **instead of telling the computer where to click, you tell it what to click on** — and it figures out the coordinates itself by looking at the screen.

---

## Real-World Applications

The same pipeline can be applied to automate any repetitive Windows workflow:
- Data entry across legacy desktop applications
- Automated testing of GUI applications without test hooks
- Screen scraping from apps that don't have APIs
- Workflow automation across multiple apps (copy from one, paste to another)
- Any task that a human would do by looking at the screen and clicking
