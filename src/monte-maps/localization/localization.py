import time
from dotenv import load_dotenv
import os
import queue
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image
import torch
import cv2
from safetensors.torch import load_file
from MegaLoc.megaloc_model import MegaLoc
import torch.nn.functional as F
import pycolmap
from filterpy.kalman import KalmanFilter


def build_pose_kalman_filter(
    initial_position,
    position_measurement_noise_std
) -> KalmanFilter:
    """
    State  : [x, y, z, vx, vy, vz]
    Measure: [x, y, z]
    """
    kalman_filter = KalmanFilter(dim_x=6, dim_z=3)

    # State transition: position += velocity * dt
    kalman_filter.F = np.eye(6)
    # Measurement matrix: position only
    kalman_filter.H = np.hstack([np.eye(3), np.zeros((3, 3))])
    # Measurement noise
    kalman_filter.R = np.eye(3) * position_measurement_noise_std ** 2
    kalman_filter.Q = np.zeros((6, 6))

    # Initial state and covariance
    # Position std ~0.5 COLMAP units (~3m)
    kalman_filter.x = np.array([*initial_position, 0.0, 0.0, 0.0])
    kalman_filter.P = np.diag([0.5, 0.5, 0.5, 0.15, 0.15, 0.15])

    return kalman_filter

class Localizer:
    def __init__(self):
        load_dotenv()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device:", self.device)

        self.initial_localization_time_seconds = float(os.getenv("INITIAL_LOCALIZATION_TIME_SECONDS", 10.0))
        self.localization_window_size_seconds = float(os.getenv("LOCALIZATION_WINDOW_SIZE_SECONDS", 5.0))
        self.colmap_to_real_world_scale = float(os.getenv("COLMAP_TO_REAL_WORLD_SCALE", 6.0))
        self.kalman_position_measurement_noise_std = float(os.getenv("KALMAN_POSITION_MEASUREMENT_NOISE_STD", 0.4))
        self.kalman_walking_acceleration_noise_std = float(os.getenv("KALMAN_WALKING_ACCELERATION_NOISE_STD", 0.3))
        image_size = int(os.getenv("IMG_SIZE", 322))

        descriptors_file_path = os.getenv("IMAGES_DESCRIPTORS", "")
        descriptors_npz = np.load(descriptors_file_path, allow_pickle=True)
        descriptors_array = descriptors_npz["descs"]
        descriptor_image_paths = [Path(p) for p in descriptors_npz["paths"]]
        self.descriptors_torch = torch.from_numpy(descriptors_array).to(self.device)
        print(f"Loaded descriptors {descriptors_array.shape} from {descriptors_file_path}")

        colmap_reconstruction_path = Path(os.getenv("COLMAP_DEMO_PROJECT", ""))
        reconstruction = pycolmap.Reconstruction(str(colmap_reconstruction_path))
        image_name_to_position = {
            image.name: image.cam_from_world().inverse().translation
            for image in reconstruction.images.values()
        }
        self.index_to_position = np.array([
            image_name_to_position.get(p.name, np.zeros(3))
            for p in descriptor_image_paths
        ], dtype=np.float32)
        print(f"Built camera positions dict for {len(self.index_to_position)} descriptors")

        self.preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        megaloc_weights_path = os.getenv("MEGALOC_WEIGHTS_PATH", "")
        self.model = MegaLoc()
        self.model.load_state_dict(load_file(megaloc_weights_path))
        self.model.eval().to(self.device)

        self.is_initializing = True
        self.initialization_start_time = None
        self.last_accepted_time = None
        self.last_predict_time = None
        self.initialization_candidate_positions = []
        self.kalman_filter = None

        self.localize_total_time_seconds = 0.0
        self.localize_call_count = 0
        self.localize_last_print_time = None

    def run(self, frame_queue, renderer, reset_localization_event):
        while True:
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                if reset_localization_event.is_set():
                    self.reset_state()
                    reset_localization_event.clear()
                continue
            if frame is None:
                break
            position = self.localize(frame)
            if position is not None:
                renderer.update_target_position(position)
            renderer.update_video_feed_preview(frame)

    def reset_state(self):
        self.is_initializing = True
        self.initialization_start_time = None
        self.last_accepted_time = None
        self.last_predict_time = None
        self.initialization_candidate_positions = []
        self.kalman_filter = None
        print("Localization state reset.")

    def retrieve_top1_index_from_frame(self, frame_bgr):
        pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        query_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            query_descriptor = F.normalize(self.model(query_tensor), p=2, dim=1)
        similarity_scores = (query_descriptor @ self.descriptors_torch.T).squeeze(0)
        return int(torch.argmax(similarity_scores).item())

    def localize(self, frame_bgr):
        """Returns a Kalman-filtered 3D position (x, y, z) or None during initialization."""
        localize_start_time = time.monotonic()
        result = self._localize(frame_bgr)
        self.localize_total_time_seconds += time.monotonic() - localize_start_time
        self.localize_call_count += 1
        if self.localize_last_print_time is None:
            self.localize_last_print_time = localize_start_time
        elif localize_start_time - self.localize_last_print_time >= 1.0:
            average_localize_time_ms = (self.localize_total_time_seconds / self.localize_call_count) * 1000
            print(f"Average prediction time: {average_localize_time_ms:.1f}ms")
            self.localize_total_time_seconds = 0.0
            self.localize_call_count = 0
            self.localize_last_print_time = localize_start_time
        return result

    def _localize(self, frame_bgr):
        current_time = time.monotonic()
        top1_index = self.retrieve_top1_index_from_frame(frame_bgr)
        candidate_position = self.index_to_position[top1_index]

        if self.is_initializing:
            if self.initialization_start_time is None:
                self.initialization_start_time = current_time
            self.initialization_candidate_positions.append(candidate_position)
            self.last_accepted_time = current_time
            elapsed = current_time - self.initialization_start_time

            if elapsed >= self.initial_localization_time_seconds:
                median_position = np.median(self.initialization_candidate_positions, axis=0)
                self.kalman_filter = build_pose_kalman_filter(
                    initial_position=median_position,
                    position_measurement_noise_std=self.kalman_position_measurement_noise_std
                )
                self.last_predict_time = current_time
                self.initialization_candidate_positions = []
                self.is_initializing = False
                print(f"Kalman filter initialized at position {median_position}")
            return None

        time_since_last_accepted = current_time - self.last_accepted_time
        if time_since_last_accepted > self.localization_window_size_seconds:
            print(f"Signal lost... Stay put and look around for {self.initial_localization_time_seconds}s.")
            self.is_initializing = True
            self.initialization_start_time = None
            self.initialization_candidate_positions = []
            self.last_predict_time = None
            self.kalman_filter = None
            return None

        delta_predict_time = current_time - self.last_predict_time
        dt = delta_predict_time
        self.kalman_filter.F[:3, 3:] = np.eye(3) * dt
        q = self.kalman_walking_acceleration_noise_std ** 2
        self.kalman_filter.Q = q * np.block([
            [0.25 * dt**4 * np.eye(3), 0.5 * dt**3 * np.eye(3)],
            [0.5  * dt**3 * np.eye(3), dt**2       * np.eye(3)],
        ])
        self.kalman_filter.predict()
        self.last_predict_time = current_time

        innovation_covariance = self.kalman_filter.H @ self.kalman_filter.P @ self.kalman_filter.H.T + self.kalman_filter.R
        innovation = candidate_position - self.kalman_filter.H @ self.kalman_filter.x
        mahalanobis_distance_squared = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
        # chi-squared thresholdfor k=3 at 95% confidence = 7.815
        if mahalanobis_distance_squared > 7.815:
            euclidean_distance_meters = np.linalg.norm(innovation) * self.colmap_to_real_world_scale
            print(f"Rejected: Mahalanobis distance squared {mahalanobis_distance_squared:.2f} > 7.815 (euclidean distance {euclidean_distance_meters:.2f}m)")
            return self.kalman_filter.x[:3].copy()

        self.last_accepted_time = current_time
        self.kalman_filter.update(candidate_position)
        return self.kalman_filter.x[:3].copy()
