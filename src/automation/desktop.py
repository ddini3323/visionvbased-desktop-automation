from __future__ import annotations

import ctypes
import io
import logging
import time

import pyautogui
import pyperclip
from PIL import Image

logger = logging.getLogger(__name__)


class DesktopController:
    def __init__(self) -> None:
        # Fix DPI scaling so pyautogui coords match screen pixels at 1920x1080
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.1

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    def take_screenshot(self) -> Image.Image:
        """Capture the primary monitor and return a PIL Image."""
        try:
            import mss
            import mss.tools

            with mss.mss() as sct:
                monitor = sct.monitors[1]  # primary monitor
                sct_img = sct.grab(monitor)
                return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        except ImportError:
            logger.warning("mss not available, falling back to pyautogui screenshot")
            return pyautogui.screenshot()

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def click(self, x: int, y: int) -> None:
        pyautogui.click(x, y)

    def double_click(self, x: int, y: int, interval: float = 0.1) -> None:
        pyautogui.doubleClick(x, y, interval=interval)

    def right_click(self, x: int, y: int) -> None:
        pyautogui.rightClick(x, y)

    def move_to(self, x: int, y: int) -> None:
        pyautogui.moveTo(x, y, duration=0.2)

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def press_keys(self, *keys: str) -> None:
        """Press a key combination, e.g. press_keys('ctrl', 's')."""
        pyautogui.hotkey(*keys)

    def type_text(self, text: str, interval: float = 0.03) -> None:
        """Type text using pyautogui (safe for ASCII, but slow for long strings)."""
        pyautogui.typewrite(text, interval=interval)

    def type_text_clipboard(self, text: str) -> None:
        """Paste text via clipboard — handles Unicode, newlines, special chars."""
        pyperclip.copy(text)
        time.sleep(0.1)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.2)

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------

    def show_desktop(self) -> None:
        """Minimise all windows (Win+D)."""
        pyautogui.hotkey("win", "d")
        time.sleep(0.5)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def wait(self, seconds: float) -> None:
        time.sleep(seconds)

    def get_screen_size(self) -> tuple:
        return pyautogui.size()
