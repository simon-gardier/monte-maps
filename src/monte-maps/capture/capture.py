import cv2
import time
import queue
import threading
from dotenv import load_dotenv
import os
import sys

load_dotenv()

CAPTURE_FPS = int(os.getenv("CAPTURE_FPS", 10))
SIMULATION_VIDEO_PATH = os.getenv("SIMULATION_VIDEO_PATH", "")


class WebcamCapture:
    def __init__(self, device_index = 0):
        self.capture = cv2.VideoCapture(device_index)
        if not self.capture.isOpened():
            sys.exit("Cannot open webcam")
        self.capture_interval = 1.0 / CAPTURE_FPS

    def read_latest_frame(self):
        success, frame = False, None
        for _ in range(int(self.capture.get(cv2.CAP_PROP_BUFFERSIZE)) or 1):
            success, frame = self.capture.read()
            if not success:
                break
        return success, frame

    def run(self, frame_queue):
        try:
            while True:
                frame_start = time.monotonic()
                success, frame = self.read_latest_frame()
                if not success:
                    break
                frame_queue.put(frame)
                elapsed = time.monotonic() - frame_start
                sleep_duration = self.capture_interval - elapsed
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
        finally:
            self.capture.release()
            frame_queue.put(None)


class SimulationCapture:
    def __init__(self, video_path = SIMULATION_VIDEO_PATH, paused_event=None):
        self.video_path = video_path
        self.paused_event = paused_event
        self.capture_interval = 1.0 / CAPTURE_FPS

    def producer(self, frame_queue: queue.Queue):
        video_capture = cv2.VideoCapture(self.video_path)
        if not video_capture.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.video_path}")
        video_fps = video_capture.get(cv2.CAP_PROP_FPS) or CAPTURE_FPS
        wall_clock_start = time.monotonic()
        capture_count = 0
        try:
            while True:
                if self.paused_event is not None and not self.paused_event.is_set():
                    pause_start = time.monotonic()
                    self.paused_event.wait()
                    wall_clock_start += time.monotonic() - pause_start
                next_wake_time = wall_clock_start + capture_count * self.capture_interval
                sleep_duration = next_wake_time - time.monotonic()
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                video_frame_index = int((time.monotonic() - wall_clock_start) * video_fps)
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, video_frame_index)
                success, frame = video_capture.read()
                if not success:
                    break
                capture_count += 1
                frame_queue.put(frame)
        finally:
            frame_queue.put(None)
            video_capture.release()

    def run(self, frame_queue):
        simulation_producer_thread = threading.Thread(target=self.producer, args=(frame_queue,), daemon=True)
        simulation_producer_thread.start()
        simulation_producer_thread.join()
