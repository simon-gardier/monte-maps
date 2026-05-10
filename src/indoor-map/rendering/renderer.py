import time
import threading
import json
import numpy as np
import cv2
import viser
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage
import pycolmap
from dotenv import load_dotenv
from scipy.spatial.transform import Rotation
import os

load_dotenv()

LERP_ALPHA = 0.08
PREVIEW_WIDTH = 320
RENDER_INTERVAL_SECONDS = 1 / 30
WORLD_UP = np.array([0.0, -1.0, 0.0]) # COLMAP coordinates have -y as up

class Renderer:
    def __init__(self):
        colmap_reconstruction_path = os.getenv("COLMAP_DEMO_PROJECT", "")
        marker_moel_path = os.getenv("MARKER_MODEL_PATH", "")
        points_of_interest_path = os.getenv("POINTS_OF_INTEREST_PATH", "")

        reconstruction = pycolmap.Reconstruction(colmap_reconstruction_path)
        point_cloud_positions = np.array([p.xyz for p in reconstruction.points3D.values()], dtype=np.float32)
        point_cloud_colors = np.array([p.color for p in reconstruction.points3D.values()], dtype=np.uint8)
        image_id_to_position = {
            image_id: image.cam_from_world().inverse().translation
            for image_id, image in reconstruction.images.items()
        }

        self.server = viser.ViserServer()
        self.server.gui.configure_theme(
            titlebar_content=TitlebarConfig(
                image=TitlebarImage(
                    image_url_light="https://my.com.uliege.be/upload/docs/image/png/2021-03/uliege_faculte_sciencesappliquees_logo_rvb_pos.png",
                    image_url_dark="https://my.com.uliege.be/upload/docs/image/png/2021-03/uliege_faculte_sciencesappliquees_logo_rvb_pos.png",
                    image_alt="Logo",
                    href="https://my.com.uliege.be/upload/docs/image/png/2021-03/uliege_faculte_sciencesappliquees_logo_rvb_pos.png",
                ),
                buttons=(
                    TitlebarButton(
                        text="GitHub", icon="GitHub", href="https://github.com/simon-gardier/indoor-map"
                    ),
                ),
            ),
            dark_mode=True,
        )
        self.server.scene.set_up_direction("-y")
        self.server.scene.add_point_cloud(
            "/building", points=point_cloud_positions, colors=point_cloud_colors, point_shape="circle", point_size=0.001
        )

        with open(points_of_interest_path) as points_of_interest_file:
            points_of_interest = json.load(points_of_interest_file)

        for camera_id_str, label_title in points_of_interest.items():
            camera_id = int(camera_id_str)
            if camera_id not in image_id_to_position:
                continue
            label_position = image_id_to_position[camera_id]
            self.server.scene.add_label(
                f"/places/{camera_id}",
                text=label_title,
                position=tuple(label_position),
            )

        with open(marker_moel_path, "rb") as glb_file:
            glb_bytes = glb_file.read()

        self.position_marker = self.server.scene.add_glb(
            "/marker",
            glb_data=glb_bytes,
            scale=0.1,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
        )

        self.shared_target_position: np.ndarray | None = None
        self.position_lock = threading.Lock()
        self.is_localization_paused = threading.Event()
        self.is_localization_paused.clear() # starts in paused state

        placeholder_image = np.zeros((int(PREVIEW_WIDTH * 9 / 16), PREVIEW_WIDTH, 3), dtype=np.uint8)
        self.preview_image_handle = self.server.gui.add_image(placeholder_image, label="Video feed")
        self.shared_preview_frame: np.ndarray | None = None
        self.frame_lock = threading.Lock()

        self.server.on_client_connect(self.on_client_connect)

    def update_target_position(self, new_position: np.ndarray):
        with self.position_lock:
            self.shared_target_position = new_position.copy()

    def update_video_feed_preview(self, frame: np.ndarray):
        preview_height = int(frame.shape[0] * PREVIEW_WIDTH / frame.shape[1])
        low_resolution_frame = cv2.resize(frame, (PREVIEW_WIDTH, preview_height))
        rgb_frame = cv2.cvtColor(low_resolution_frame, cv2.COLOR_BGR2RGB)
        with self.frame_lock:
            self.shared_preview_frame = rgb_frame

    def compute_billboard_wxyz(self, object_position: np.ndarray, camera_position: np.ndarray) -> tuple:
        """Return a [w, x, y, z] quaternion that rotates the GLB's toward the camera,
        constrained to rotate only around the world up axis."""
        delta = camera_position - object_position
        # Project onto the horizontal plane so the model yaws
        forward_horizontal = np.array([delta[0], 0.0, delta[2]])
        forward_norm = np.linalg.norm(forward_horizontal)
        if forward_norm < 1e-6:
            return (1.0, 0.0, 0.0, 0.0)
        forward_horizontal = forward_horizontal / forward_norm

        right = np.cross(WORLD_UP, forward_horizontal)
        right_norm = np.linalg.norm(right)
        right = np.array([1.0, 0.0, 0.0]) if right_norm < 1e-6 else right / right_norm

        actual_up = np.cross(forward_horizontal, right)
        rotation_matrix = np.column_stack([right, actual_up, forward_horizontal])
        q = Rotation.from_matrix(rotation_matrix).as_quat()
        return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))

    def on_client_connect(self, client: viser.ClientHandle):
        play_pause_button = client.gui.add_button("⏯️ Start")

        @play_pause_button.on_click
        def _(_):
            if self.is_localization_paused.is_set():
                self.is_localization_paused.clear()
                play_pause_button.label = "⏯️ Resume"
            else:
                self.is_localization_paused.set()
                play_pause_button.label = "⏯️ Pause"

    def run(self):
        current_display_position = np.array([0.0, 0.0, 0.0])
        while True:
            start = time.monotonic()
            with self.position_lock:
                target = self.shared_target_position
            if target is not None:
                current_display_position = current_display_position + LERP_ALPHA * (target - current_display_position)
                self.position_marker.position = tuple(current_display_position)

            connected_clients = self.server.get_clients()
            if connected_clients:
                first_client = next(iter(connected_clients.values()))
                camera_position = np.array(first_client.camera.position)
                self.position_marker.wxyz = self.compute_billboard_wxyz(current_display_position, camera_position)

            with self.frame_lock:
                preview_frame = self.shared_preview_frame
            if preview_frame is not None:
                self.preview_image_handle.image = preview_frame

            elapsed = time.monotonic() - start
            sleep_time = RENDER_INTERVAL_SECONDS - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
