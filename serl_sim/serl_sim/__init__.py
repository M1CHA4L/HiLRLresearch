import os
import sys

# Headless Linux (e.g. the A100 server): render offscreen with EGL. Must run before `import mujoco`.
if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
