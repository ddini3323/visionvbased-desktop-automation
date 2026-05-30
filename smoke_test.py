import os, sys, time
sys.stdout.reconfigure(encoding=utf-8)
sys.path.insert(0, .)
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get(ANTHROPIC_AUTH_TOKEN) or os.environ.get(ANTHROPIC_API_KEY, )
base_url = os.environ.get(ANTHROPIC_BASE_URL)
model = os.environ.get(ANTHROPIC_MODEL, claude-sonnet-latest)

from src.automation.desktop import DesktopController
from src.grounding.hybrid_grounder import HybridGrounder

desktop = DesktopController()
grounder = HybridGrounder(api_key=api_key, base_url=base_url, model=model)

print(Showing desktop...)
desktop.show_desktop()
time.sleep(1.5)

print(Taking screenshot...)
screenshot = desktop.take_screenshot()
print(fScreenshot: {screenshot.size})

print(Running hybrid cascade...)
t0 = time.perf_counter()
result = grounder.ground(screenshot, Notepad shortcut icon on the Windows desktop)
elapsed = time.perf_counter() - t0

print(ffound={result.found})
print(fx={result.x}  y={result.y})
print(fconfidence={result.confidence:.2f})
print(fmethod={result.method})
print(ftotal_time={elapsed:.2f}s)
print(freasoning={result.reasoning})

if result.found:
    annotated = grounder.annotate(screenshot, result)
    annotated.save(test_detection.png)
    print(Saved test_detection.png)
else:
    screenshot.save(debug_screenshot.png)
    print(ICON NOT FOUND -- saved debug_screenshot.png)

