"""ScreenSeekeR-inspired visual grounding for desktop icon detection.

Implements the core algorithm from:
  "ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use"
  arXiv:2504.07981 — Li et al., NUS / HKBU

The original paper uses GPT-4o as the planner and OS-Atlas-7B as the
specialized grounding model.  Here we use Claude for both roles, which is a
valid adaptation: Claude has strong vision capabilities and can predict
normalized bounding boxes given a text description.

Algorithm (visual_search):
  1. Termination: if depth >= max_depth or crop <= min_crop_size, call the
     grounder directly (try_ground_in_patch).
  2. Position inference: send the crop + POSITION_INFERENCE_PROMPT to Claude.
     The model returns descriptions wrapped in <element>, <area>, <neighbor>
     XML tags identifying where the target is likely to be.
  3. Ground each candidate description + the original instruction against the
     crop to collect bounding boxes (votes).
  4. Score candidate patches with a Gaussian kernel (sigma=0.3): each vote
     contributes exp(-d²/(2σ²)) to the patch it falls inside.
  5. Non-Maximum Suppression at IoU ≥ 0.5 on scored patches.
  6. Recurse into top-scoring patches depth-first until a match is confirmed
     by the existence check.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import re
import time
from typing import List, Optional, Tuple

import anthropic
from PIL import Image, ImageDraw

from .base import BaseGrounder, GroundingResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates (exact text from arXiv:2504.07981 source code)
# ---------------------------------------------------------------------------

POSITION_INFERENCE_PROMPT = """I want to identify a UI element that best matches my instruction. Please help me determine which region(s) of the screenshot to focus on and list the UI elements that might appear next to the target.
If the target does not exist in the screenshot, please output 'No target'.

**Output Requirements:**
1. List the possible regions in descending order of probability.
2. **Always make specific, clear and unique references to avoid ambiguity**. References such as 'Other icons' and 'window' are NOT allowed, because they don't refer to a specific UI element.
3. Use the following XML tags to describe items in the screenshot:
   - `<element>`: Wrap a specific UI element.
   - `<area>`: Describe an area of the UI containing multiple elements.
   - `<neighbor>`: Describe a UI element that may appear around the target, to help anchor its location.

**Example Output (No need to follow the strict sentence format):**
1. The <element>shortcut link</element> is most likely to be found in the <area>Settings window</area>, in the <area>tools panel in settings window</area> next to the <neighbor>Search button in the settings window</neighbor>. There is also a <neighbor>update button in the settings window</neighbor> nearby.
2. The target may also appear in the <area>Web Browser</area>, in the <area>Bookmark bar in browser</area>, next to the <neighbor>Search button in the bookmark bar in web browser</neighbor> and the <neighbor>open bookmark collection button in the browser</neighbor>.
...

**Important Notes:**
- The target UI element is guaranteed to be present in the screenshot. Do **not** speculate about any operations that could change the screenshot, such as navigating to another page or opening a menu.

**Instruction:**
{instruction}"""

EXISTENCE_PROMPT = """You are given a cropped screenshot. Your task is to evaluate whether the marked element in red box matches the target described in my instruction.

Please follow these steps:
1. Analyze the screenshot by describing its visible content and functionalities.
2. Determine which of the following applies:
    - 'is_target': The marked element is the target.
    - 'target_elsewhere': The marked element is not the target, but the target exists elsewhere in the screenshot.
    - 'target_not_found': The marked element is not the target, and the target does not exist in the screenshot.
3. If the target exists, rewrite the instruction to make it more specific and clear. This should include unambiguous details like labels, text, or position.

After your analysis, provide the result in JSON format with the following fields:
- 'result': (str) One of 'is_target', 'target_elsewhere', or 'target_not_found'.
- 'new_instruction': (str, default null) A clearer, more specific version of the instruction, if applicable.

Here is my instruction:
{instruction}"""

GROUNDER_PROMPT = """Locate the UI element described below in this screenshot.
Return ONLY a JSON object on a single line with no other text:
{{"found": <bool>, "x1": <float 0-1>, "y1": <float 0-1>, "x2": <float 0-1>, "y2": <float 0-1>, "confidence": <float 0-1>}}

All coordinates are normalized to [0, 1] relative to the image dimensions.
(x1,y1) is the top-left corner; (x2,y2) is the bottom-right corner.

Element to locate: {description}"""

PLANNER_SYSTEM = (
    "You are an expert in using electronic devices and interacting with "
    "graphic interfaces."
)
GROUNDER_SYSTEM = (
    "You are a precise GUI element locator. You identify UI elements in "
    "screenshots and return their normalized bounding box coordinates as JSON."
)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

NormBox = Tuple[float, float, float, float]   # x1,y1,x2,y2 in [0,1]


def _iou(a: NormBox, b: NormBox) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def _nms(boxes: List[Tuple[float, NormBox]], iou_threshold: float = 0.5) -> List[NormBox]:
    """Non-Maximum Suppression. `boxes` is a list of (score, norm_box)."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x[0], reverse=True)
    kept: List[NormBox] = []
    for score, box in boxes:
        if all(_iou(box, k) < iou_threshold for k in kept):
            kept.append(box)
    return kept


def _center(box: NormBox) -> Tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _gaussian_score(vote_center: Tuple[float, float], patch: NormBox, sigma: float = 0.3) -> float:
    """Score of a vote for a patch using a Gaussian kernel (sigma in [0,1] space)."""
    px1, py1, px2, py2 = patch
    vx, vy = vote_center
    # Check if vote center is inside the patch
    if not (px1 <= vx <= px2 and py1 <= vy <= py2):
        return 0.0
    # Normalize vote position within the patch to [0,1]
    pw = px2 - px1 or 1e-9
    ph = py2 - py1 or 1e-9
    rel_x = (vx - px1) / pw - 0.5   # relative to patch centre
    rel_y = (vy - py1) / ph - 0.5
    dist_sq = rel_x ** 2 + rel_y ** 2
    return math.exp(-dist_sq / (2 * sigma ** 2))


def _score_patches(patches: List[NormBox], votes: List[NormBox], sigma: float = 0.3) -> List[float]:
    scores = []
    for patch in patches:
        patch_score = sum(_gaussian_score(_center(v), patch, sigma) for v in votes)
        scores.append(patch_score)
    return scores


def _view_to_real(view_box: NormBox, viewport: NormBox) -> NormBox:
    """Map a normalized box within a cropped viewport back to full-image coordinates."""
    vx1, vy1, vx2, vy2 = viewport
    vw, vh = vx2 - vx1, vy2 - vy1
    return (
        vx1 + view_box[0] * vw,
        vy1 + view_box[1] * vh,
        vx1 + view_box[2] * vw,
        vy1 + view_box[3] * vh,
    )


def _auto_dilate(
    image: Image.Image,
    box: NormBox,
    min_size: Tuple[int, int] = (1280, 720),
    max_ratio: float = 3.0,
) -> List[NormBox]:
    """Expand a small or extreme-aspect-ratio box to at least min_size.

    Returns one or more NormBox objects (split when aspect ratio is extreme).
    """
    iw, ih = image.size
    bw = max((box[2] - box[0]) * iw, 1)
    bh = max((box[3] - box[1]) * ih, 1)

    # If already large enough and not extreme, return as-is
    if bw >= min_size[0] and bh >= min_size[1] and bw / bh <= max_ratio and bh / bw <= max_ratio:
        return [box]

    # Pad to min_size
    target_w = max(bw, min_size[0])
    target_h = max(bh, min_size[1])
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    half_w = target_w / iw / 2
    half_h = target_h / ih / 2
    dilated: NormBox = (
        max(0.0, cx - half_w),
        max(0.0, cy - half_h),
        min(1.0, cx + half_w),
        min(1.0, cy + half_h),
    )

    dw = (dilated[2] - dilated[0]) * iw
    dh = (dilated[3] - dilated[1]) * ih

    # Split if still extreme aspect ratio
    if dw / dh > max_ratio:
        mid = (dilated[0] + dilated[2]) / 2
        return [
            (dilated[0], dilated[1], mid, dilated[3]),
            (mid, dilated[1], dilated[2], dilated[3]),
        ]
    if dh / dw > max_ratio:
        mid = (dilated[1] + dilated[3]) / 2
        return [
            (dilated[0], dilated[1], dilated[2], mid),
            (dilated[0], mid, dilated[2], dilated[3]),
        ]

    return [dilated]


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _pil_to_b64(image: Image.Image) -> str:
    """Encode PIL image as base64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def _crop_norm(image: Image.Image, box: NormBox) -> Image.Image:
    w, h = image.size
    x1 = int(box[0] * w)
    y1 = int(box[1] * h)
    x2 = int(box[2] * w)
    y2 = int(box[3] * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image.crop((x1, y1, x2, y2))


def _annotate_with_red_box(image: Image.Image, box_norm: NormBox) -> Image.Image:
    """Draw a red rectangle on a copy of image for existence verification."""
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    w, h = out.size
    x1 = int(box_norm[0] * w)
    y1 = int(box_norm[1] * h)
    x2 = int(box_norm[2] * w)
    y2 = int(box_norm[3] * h)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=5)
    return out


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "")
    try:
        return __import__("json").loads(text.strip())
    except Exception:
        pass
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return __import__("json").loads(m.group())
        except Exception:
            pass
    return {}


def _extract_xml_tags(text: str) -> List[str]:
    """Extract all text inside <element>, <area>, <neighbor> tags."""
    pattern = r"<(?:element|area|neighbor)>(.*?)</(?:element|area|neighbor)>"
    return re.findall(pattern, text, re.DOTALL)


# ---------------------------------------------------------------------------
# VLMGrounder
# ---------------------------------------------------------------------------

class VLMGrounder(BaseGrounder):
    """ScreenSeekeR-inspired VLM grounding using recursive visual search."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        max_retries: int = 3,
        max_depth: int = 3,
        min_crop_px: Tuple[int, int] = (640, 360),
        base_url: str = None,
    ) -> None:
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**kwargs)
        self._model = model
        self._max_retries = max_retries
        self._max_depth = max_depth
        self._min_crop_px = min_crop_px

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ground(self, screenshot: Image.Image, target: str) -> GroundingResult:
        orig_w, orig_h = screenshot.size
        full_viewport: NormBox = (0.0, 0.0, 1.0, 1.0)

        result_box = self._visual_search(screenshot, target, full_viewport, depth=0)

        if result_box is None:
            logger.warning("ScreenSeekeR: all strategies exhausted, target not found.")
            return GroundingResult(
                x=orig_w // 2,
                y=orig_h // 2,
                confidence=0.0,
                bbox=(0, 0, orig_w, orig_h),
                method="screenseeker",
                found=False,
                reasoning="Recursive visual search failed to locate the target.",
            )

        norm_box, confidence, reasoning = result_box
        cx = int(((norm_box[0] + norm_box[2]) / 2) * orig_w)
        cy = int(((norm_box[1] + norm_box[3]) / 2) * orig_h)
        bbox = (
            int(norm_box[0] * orig_w),
            int(norm_box[1] * orig_h),
            int(norm_box[2] * orig_w),
            int(norm_box[3] * orig_h),
        )
        return GroundingResult(
            x=cx,
            y=cy,
            confidence=confidence,
            bbox=bbox,
            method="screenseeker",
            found=True,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Recursive visual search (core ScreenSeekeR algorithm)
    # ------------------------------------------------------------------

    def _visual_search(
        self,
        image: Image.Image,
        target: str,
        viewport: NormBox,
        depth: int,
    ) -> Optional[Tuple[NormBox, float, str]]:
        """Recursive search. Returns (full_image_norm_box, confidence, reasoning) or None."""
        iw, ih = image.size

        # Termination condition: image is small enough to ground directly
        if depth >= self._max_depth or (iw <= self._min_crop_px[0] and ih <= self._min_crop_px[1]):
            return self._try_ground_in_patch(image, target, viewport)

        # --- Step 2: Position inference ---
        planner_text = self._call_planner(image, target)
        if planner_text is None:
            return self._try_ground_in_patch(image, target, viewport)

        if "No target" in planner_text:
            logger.debug("Planner says 'No target' at depth %d", depth)
            return self._try_ground_in_patch(image, target, viewport)

        # --- Step 3: Extract candidate descriptions ---
        candidates = _extract_xml_tags(planner_text)
        if not candidates:
            return self._try_ground_in_patch(image, target, viewport)

        logger.debug("Depth %d: %d candidate descriptions from planner", depth, len(candidates))

        # --- Step 4: Ground candidates to collect votes ---
        votes: List[NormBox] = []

        # Always ground the original instruction directly (view_direct_bbox)
        direct = self._call_grounder(image, target)
        if direct:
            votes.append(direct)

        # Ground each candidate description
        for desc in candidates[:8]:  # limit to first 8 to control API cost
            box = self._call_grounder(image, desc)
            if box:
                votes.append(box)

        if not votes:
            return self._try_ground_in_patch(image, target, viewport)

        # --- Step 5: Auto-dilate small patches ---
        patches: List[NormBox] = []
        for v in votes:
            patches.extend(_auto_dilate(image, v, self._min_crop_px))

        # --- Step 6: Score patches ---
        scores = _score_patches(patches, votes, sigma=0.3)

        # --- Step 7: NMS ---
        scored_patches = list(zip(scores, patches))
        kept = _nms(scored_patches, iou_threshold=0.5)

        if not kept:
            return self._try_ground_in_patch(image, target, viewport)

        # --- Step 8: Recurse into top patches ---
        for patch in kept[:3]:   # try top 3 NMS survivors
            sub_viewport = _view_to_real(patch, viewport)
            sub_image = _crop_norm(image, patch)

            if sub_image.size[0] < 10 or sub_image.size[1] < 10:
                continue

            result = self._visual_search(sub_image, target, sub_viewport, depth + 1)
            if result is not None:
                return result

        return None

    # ------------------------------------------------------------------
    # Leaf grounding: try to ground directly in the current crop
    # ------------------------------------------------------------------

    def _try_ground_in_patch(
        self, image: Image.Image, target: str, viewport: NormBox
    ) -> Optional[Tuple[NormBox, float, str]]:
        box = self._call_grounder(image, target)
        if box is None:
            return None

        # Map the in-crop normalized box back to full-image coordinates
        real_box = _view_to_real(box, viewport)

        # Existence verification: draw red bbox on crop and ask Claude to confirm
        confirmed, reasoning = self._check_existence(image, box, target)
        if confirmed:
            return real_box, 0.9, reasoning
        if reasoning == "target_elsewhere":
            # Target is visible in the crop but our box is wrong — try again once
            box2 = self._call_grounder(image, target)
            if box2:
                real_box2 = _view_to_real(box2, viewport)
                confirmed2, reasoning2 = self._check_existence(image, box2, target)
                if confirmed2:
                    return real_box2, 0.85, reasoning2
        return None

    # ------------------------------------------------------------------
    # VLM call helpers
    # ------------------------------------------------------------------

    def _call_planner(self, image: Image.Image, target: str) -> Optional[str]:
        """Call Claude with the ScreenSeekeR position inference prompt."""
        prompt = POSITION_INFERENCE_PROMPT.format(instruction=target)
        b64 = _pil_to_b64(image)

        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=512,
                    system=PLANNER_SYSTEM,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": b64,
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                )
                return response.content[0].text
            except anthropic.RateLimitError:
                wait = 2.0 ** attempt
                logger.warning("Rate limit; waiting %.1fs (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
            except Exception as exc:
                logger.warning("Planner call failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(0.5)
        return None

    def _call_grounder(self, image: Image.Image, description: str) -> Optional[NormBox]:
        """Ask Claude to predict a normalized bounding box for `description`."""
        prompt = GROUNDER_PROMPT.format(description=description)
        b64 = _pil_to_b64(image)

        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=128,
                    system=GROUNDER_SYSTEM,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": b64,
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                )
                data = _extract_json(response.content[0].text)
                if not data.get("found", True) is False:
                    # Parse normalized coordinates
                    x1 = float(data.get("x1", 0))
                    y1 = float(data.get("y1", 0))
                    x2 = float(data.get("x2", 1))
                    y2 = float(data.get("y2", 1))
                    # Clamp and validate
                    x1, x2 = sorted([max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))])
                    y1, y2 = sorted([max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))])
                    if x2 - x1 > 0.001 and y2 - y1 > 0.001:
                        return (x1, y1, x2, y2)
                return None
            except anthropic.RateLimitError:
                time.sleep(2.0 ** attempt)
            except Exception as exc:
                logger.warning("Grounder call failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(0.5)
        return None

    def _check_existence(
        self, image: Image.Image, box: NormBox, target: str
    ) -> Tuple[bool, str]:
        """Verify that `box` contains `target` using the existence prompt."""
        annotated = _annotate_with_red_box(image, box)
        prompt = EXISTENCE_PROMPT.format(instruction=target)
        b64 = _pil_to_b64(annotated)

        for attempt in range(self._max_retries):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=256,
                    system=PLANNER_SYSTEM,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": b64,
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                )
                data = _extract_json(response.content[0].text)
                result = data.get("result", "target_not_found")
                is_target = result == "is_target"
                return is_target, result
            except anthropic.RateLimitError:
                time.sleep(2.0 ** attempt)
            except Exception as exc:
                logger.warning("Existence check failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(0.5)

        # On failure, trust the grounder result (optimistic)
        return True, "existence_check_failed"
