import argparse
import threading

from localization.localization import Localizer
from capture.capture import WebcamCapture, SimulationCapture
from rendering.renderer import Renderer


def localization_thread_target(frame_generator, localizer: Localizer, renderer: Renderer):
    for frame in frame_generator:
        renderer.update_video_feed_preview(frame)
        position = localizer.localize(frame)
        if position is not None:
            renderer.update_target_position(position)


def main():
    parser = argparse.ArgumentParser(description="Indoor Maps")
    parser.add_argument("--mode", choices=["webcam", "simulation"], default="simulation")
    args = parser.parse_args()

    renderer = Renderer()

    if args.mode == "webcam":
        frame_generator = WebcamCapture().frames()
    else:
        frame_generator = SimulationCapture(paused_event=renderer.is_localization_paused).frames()

    localizer = Localizer()
    localization_thread = threading.Thread(
        target=localization_thread_target,
        args=(frame_generator, localizer, renderer),
        daemon=True,
    )
    localization_thread.start()

    renderer.run_render_loop()


if __name__ == "__main__":
    main()
