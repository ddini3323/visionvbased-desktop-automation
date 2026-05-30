from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pyautogui
import pyperclip
import win32gui

from .desktop import DesktopController

logger = logging.getLogger(__name__)

# Notepad window title substrings (works for both Win10 and Win11 Notepad)
_NOTEPAD_TITLES = ("Notepad", "Kladblok", "Blocco note", "Bloc-notes")


def _is_notepad_window(title: str) -> bool:
    return any(t.lower() in title.lower() for t in _NOTEPAD_TITLES)


class NotepadController:
    def __init__(self, desktop: DesktopController, grounder) -> None:
        self._desktop = desktop
        self._grounder = grounder

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------

    def launch(self, max_attempts: int = 3) -> bool:
        """Ground the Notepad icon and double-click to open it."""
        for attempt in range(1, max_attempts + 1):
            logger.info("Launch attempt %d/%d", attempt, max_attempts)

            self._desktop.show_desktop()
            self._desktop.wait(0.5)

            screenshot = self._desktop.take_screenshot()
            result = self._grounder.ground(screenshot, "Notepad shortcut icon on the Windows desktop")

            if not result.found:
                logger.warning("Notepad icon not found on attempt %d", attempt)
                if attempt < max_attempts:
                    time.sleep(1.0)
                continue

            logger.info(
                "Icon found at (%d, %d) confidence=%.2f method=%s",
                result.x, result.y, result.confidence, result.method,
            )
            self._desktop.double_click(result.x, result.y)

            if self.wait_for_window(timeout=10.0):
                return True

            logger.warning("Notepad did not open after attempt %d", attempt)
            time.sleep(1.0)

        return False

    # ------------------------------------------------------------------
    # Window detection
    # ------------------------------------------------------------------

    def wait_for_window(self, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._get_window_hwnd() is not None:
                return True
            time.sleep(0.5)
        return False

    def _get_window_hwnd(self) -> Optional[int]:
        found = []

        def callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if _is_notepad_window(title):
                    found.append(hwnd)

        win32gui.EnumWindows(callback, None)
        return found[0] if found else None

    def is_open(self) -> bool:
        return self._get_window_hwnd() is not None

    def focus_window(self) -> bool:
        hwnd = self._get_window_hwnd()
        if hwnd is None:
            return False
        try:
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.3)
            return True
        except Exception as exc:
            logger.warning("SetForegroundWindow failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Content
    # ------------------------------------------------------------------

    def type_content(self, content: str) -> None:
        """Focus Notepad text area and paste content via clipboard."""
        if not self.focus_window():
            logger.warning("Could not focus Notepad window")
            return

        # Click centre of the text area (safe approximation)
        hwnd = self._get_window_hwnd()
        if hwnd:
            try:
                rect = win32gui.GetWindowRect(hwnd)
                cx = (rect[0] + rect[2]) // 2
                cy = (rect[1] + rect[3]) // 2
                self._desktop.click(cx, cy)
            except Exception:
                pass

        time.sleep(0.2)
        self._desktop.type_text_clipboard(content)
        time.sleep(0.3)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_file(self, filename: str, save_dir: Path) -> Path:
        """Save the current document via Ctrl+S → Save As dialog."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        full_path = save_dir / filename

        self.focus_window()
        pyautogui.hotkey("ctrl", "s")
        time.sleep(1.2)  # wait for Save As dialog

        # Clear filename field and type full absolute path
        pyautogui.hotkey("ctrl", "a")
        time.sleep(0.1)

        pyperclip.copy(str(full_path))
        time.sleep(0.1)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.2)

        pyautogui.press("enter")
        time.sleep(0.8)

        # If a "file already exists – overwrite?" dialog appears, press Enter (Yes)
        self._dismiss_overwrite_dialog()

        return full_path

    def _dismiss_overwrite_dialog(self) -> None:
        """Dismiss 'file already exists' confirmation dialogs if present."""
        time.sleep(0.3)
        hwnd_list = []

        def cb(hwnd, _):
            title = win32gui.GetWindowText(hwnd)
            if "Confirm" in title or "Save As" in title or "Replace" in title:
                hwnd_list.append(hwnd)

        win32gui.EnumWindows(cb, None)
        if hwnd_list:
            pyautogui.press("enter")
            time.sleep(0.3)

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self) -> bool:
        """Close Notepad, discarding any unsaved changes."""
        if not self.is_open():
            return True

        self.focus_window()
        pyautogui.hotkey("alt", "f4")
        time.sleep(0.6)

        # Handle "Do you want to save?" dialog
        if self._check_save_dialog():
            # Press the "Don't Save" button (Alt+N on en-US Notepad)
            pyautogui.press("n")
            time.sleep(0.4)

        # Fallback: dismiss any remaining dialog with Enter
        if self._check_save_dialog():
            pyautogui.press("enter")
            time.sleep(0.4)

        deadline = time.time() + 3.0
        while time.time() < deadline:
            if not self.is_open():
                return True
            time.sleep(0.3)

        logger.warning("Notepad did not close within timeout")
        return False

    def _check_save_dialog(self) -> bool:
        """Return True if a save-changes dialog is visible."""
        dialogs = []

        def cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if "Notepad" in title or "Save" in title:
                    dialogs.append(hwnd)

        win32gui.EnumWindows(cb, None)
        # If there's more than one Notepad-related window, the extra one is the dialog
        notepad_wins = [h for h in dialogs if _is_notepad_window(win32gui.GetWindowText(h))]
        return len(notepad_wins) > 1 or len(dialogs) > len(notepad_wins)
