import argparse
import queue
import threading
import time

from localization.localization import Localizer
from capture.capture import WebcamCapture, SimulationCapture
from rendering.renderer import Renderer


def main():
    parser = argparse.ArgumentParser(description="Indoor Maps")
    parser.add_argument("--mode", choices=["webcam", "simulation"], default="simulation")
    args = parser.parse_args()

    renderer = Renderer()
    localizer = Localizer()
    frame_queue = queue.Queue(maxsize=2)

    if args.mode == "webcam":
        capture = WebcamCapture(2, paused_event=renderer.is_localization_paused)
    else:
        capture = SimulationCapture(paused_event=renderer.is_localization_paused)

    capture_thread = threading.Thread(target=capture.run, args=(frame_queue,), daemon=True)
    localization_thread = threading.Thread(target=localizer.run, args=(frame_queue, renderer, renderer.reset_localization_event), daemon=True)
    renderer_thread = threading.Thread(target=renderer.run, daemon=True)

    capture_thread.start()
    localization_thread.start()
    renderer_thread.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
