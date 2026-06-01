"""Desktop automation entry point â€” VLM visual grounding + Notepad workflow."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.api import JSONPlaceholderClient
from src.automation import DesktopController, NotepadController
from src.grounding import HybridGrounder
from src.utils import FileManager, setup_logger


def main() -> bool:
    # ------------------------------------------------------------------ setup
    log_dir = Path(os.environ.get("LOG_DIR", "logs"))
    log_dir.mkdir(exist_ok=True)
    logger = setup_logger("main", log_file=str(log_dir / "automation.log"), level=logging.INFO)
    logger.info("=" * 60)
    logger.info("Desktop Automation â€” VLM Visual Grounding")
    logger.info("=" * 60)

    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        logger.error("ANTHROPIC_API_KEY not set. Copy .env.example â†’ .env and add your key.")
        sys.exit(1)

    model = os.environ.get("VLM_MODEL", "claude-sonnet-4-6")
    max_retries = int(os.environ.get("VLM_MAX_RETRIES", "3"))

    # ------------------------------------------------------------------ init
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    grounder = HybridGrounder(api_key=api_key, model=model, base_url=base_url, max_retries=max_retries)
    desktop = DesktopController()
    notepad = NotepadController(desktop=desktop, grounder=grounder)
    api_client = JSONPlaceholderClient()
    file_manager = FileManager(os.environ.get("PROJECT_DIR_NAME", "tjm-project"))

    screenshot_dir = Path(os.environ.get("SCREENSHOT_DIR", "screenshots"))
    screenshot_dir.mkdir(exist_ok=True)
    project_dir = file_manager.setup_project_dir()
    logger.info("Saving posts to: %s", project_dir)

    # ------------------------------------------------------------------ fetch
    logger.info("Fetching posts from JSONPlaceholder API...")
    posts = api_client.fetch_posts(limit=10)
    logger.info("Got %d posts", len(posts))

    results: dict = {"success": 0, "failed": 0, "files": []}
    start_time = time.time()

    # ------------------------------------------------------------------ loop
    for i, post in enumerate(posts, 1):
        logger.info("")
        logger.info("â”€" * 50)
        logger.info("Post %d/10  id=%d  title=%s", i, post.id, post.title[:60])
        logger.info("â”€" * 50)

        try:
            desktop.show_desktop()
            desktop.wait(0.5)

            # Ground the Notepad icon
            screenshot = desktop.take_screenshot()
            result = grounder.ground(screenshot, "Notepad shortcut icon on the Windows desktop")

            if not result.found:
                logger.error("Notepad icon not found (post %d) â€” skipping", post.id)
                results["failed"] += 1
                continue

            logger.info(
                "Icon found at (%d, %d)  confidence=%.2f  method=%s",
                result.x, result.y, result.confidence, result.method,
            )

            # Save annotated screenshot
            annotated = grounder.annotate(screenshot, result)
            ann_path = screenshot_dir / f"post_{post.id:02d}_detection.png"
            annotated.save(str(ann_path))
            logger.info("Annotated screenshot â†’ %s", ann_path)

            # Launch Notepad
            desktop.double_click(result.x, result.y)
            if not notepad.wait_for_window(timeout=10.0):
                logger.error("Notepad did not open (post %d) â€” skipping", post.id)
                results["failed"] += 1
                continue

            logger.info("Notepad opened")
            desktop.wait(0.5)

            # Type content
            content = file_manager.format_content(post)
            notepad.type_content(content)
            desktop.wait(0.3)

            # Save file
            save_path = file_manager.get_save_path(post)
            save_path = file_manager.handle_existing(save_path)
            saved = notepad.save_file(save_path.name, save_path.parent)
            results["files"].append(str(saved))
            logger.info("Saved â†’ %s", saved)

            # Close
            notepad.close()
            desktop.wait(1.0)

            results["success"] += 1

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            try:
                notepad.close()
            except Exception:
                pass
            break
        except Exception as exc:
            logger.error("Unexpected error on post %d: %s", post.id, exc, exc_info=True)
            results["failed"] += 1
            try:
                notepad.close()
            except Exception:
                pass
            desktop.wait(1.0)

    # ------------------------------------------------------------------ summary
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("DONE  success=%d  failed=%d  time=%.1fs", results["success"], results["failed"], elapsed)
    logger.info("Files saved to: %s", project_dir)
    for f in results["files"]:
        logger.info("  %s", f)
    logger.info("=" * 60)

    return results["failed"] == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
