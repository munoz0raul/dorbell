"""20-second NPU FPS measurement."""
import sys, time, numpy as np, collections

sys.argv = ["dp.py", "--backend", "npu"]
import cv2
cv2.namedWindow = lambda *a, **kw: None
cv2.setWindowProperty = lambda *a, **kw: None
cv2.imshow = lambda *a, **kw: None
cv2.destroyAllWindows = lambda: None

import importlib.util
spec = importlib.util.spec_from_file_location("db", "/home/ubuntu/doorbell_prototype.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

_ready = {}
npu = mod.NpuFaceEngine(on_ready=lambda: _ready.update(npu=True))
npu.start()

for _ in range(30):
    if _ready.get("npu"):
        break
    time.sleep(0.2)
print("NPU ready:", _ready.get("npu"))

start = time.time()
frames_sent = 0
while time.time() - start < 20:
    frame = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    npu.submit(frame)
    frames_sent += 1
    time.sleep(1/30)

time.sleep(2)
fps = npu.fps_counter.fps()
print(f"NPU FPS={fps:.1f}  frames_sent={frames_sent}")
npu.stop()
