from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List

import requests

logger = logging.getLogger(__name__)


@dataclass
class Post:
    id: int
    userId: int
    title: str
    body: str


MOCK_POSTS: List[Post] = [
    Post(id=1, userId=1, title="Introduction to Automation", body="Automation allows us to perform repetitive tasks without manual intervention, saving time and reducing human error across many domains."),
    Post(id=2, userId=1, title="Python for Desktop Automation", body="Python is an excellent language for desktop automation due to its readable syntax and rich ecosystem of libraries like pyautogui and win32api."),
    Post(id=3, userId=1, title="Computer Vision Basics", body="Computer vision enables machines to interpret and understand visual information from the world, powering applications from face recognition to autonomous vehicles."),
    Post(id=4, userId=1, title="Visual Grounding Explained", body="Visual grounding is the ability to locate specific objects or UI elements in images based on a natural language description, bridging vision and language."),
    Post(id=5, userId=2, title="Large Language Models and GUI", body="Modern LLMs can understand screenshots and predict where to click, enabling a new generation of flexible GUI automation agents that don't rely on fixed templates."),
    Post(id=6, userId=2, title="Windows API Overview", body="The Windows API provides low-level access to the operating system, allowing developers to create, manage, and interact with windows and system resources programmatically."),
    Post(id=7, userId=2, title="Clipboard Operations in Python", body="Clipboard access via pyperclip allows Python programs to copy and paste text containing Unicode characters, line breaks, and special symbols reliably."),
    Post(id=8, userId=3, title="Retry Logic Best Practices", body="Implementing exponential backoff with jitter in retry logic prevents thundering herd problems and makes distributed systems more resilient under load."),
    Post(id=9, userId=3, title="Screenshot Capture Techniques", body="Libraries like mss capture screenshots significantly faster than PIL's ImageGrab because they use native OS APIs, making them suitable for real-time automation."),
    Post(id=10, userId=3, title="Error Handling in Automation", body="Robust error handling in automation scripts should log failures, attempt recovery, and degrade gracefully so that a single failure doesn't stop the entire workflow."),
]


class JSONPlaceholderClient:
    BASE_URL = "https://jsonplaceholder.typicode.com"

    def fetch_posts(self, limit: int = 10) -> List[Post]:
        """Fetch posts from the API with retry logic, falling back to mock data."""
        delays = [1.0, 2.0, 4.0]
        for attempt, delay in enumerate(delays, 1):
            try:
                resp = requests.get(
                    f"{self.BASE_URL}/posts",
                    params={"_limit": limit},
                    timeout=30,
                )
                resp.raise_for_status()
                raw = resp.json()
                posts = [
                    Post(
                        id=item["id"],
                        userId=item["userId"],
                        title=item["title"],
                        body=item["body"],
                    )
                    for item in raw[:limit]
                ]
                logger.info("Fetched %d posts from API", len(posts))
                return posts
            except Exception as exc:
                logger.warning("API attempt %d failed: %s", attempt, exc)
                if attempt < len(delays):
                    time.sleep(delay)

        logger.warning("All API attempts failed — using mock data")
        return MOCK_POSTS[:limit]
