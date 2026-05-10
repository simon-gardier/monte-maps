import time
from dotenv import load_dotenv
import os
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
    position_measurement_noise_std,
    walking_acceleration_noise_std,
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
    # Process noise: CWNA model, axes independent, state order [x,y,z,vx,vy,vz]
    q = walking_acceleration_noise_std ** 2
    kalman_filter.Q = q * np.block([
        [0.25 * np.eye(3), 0.5 * np.eye(3)],
        [0.5  * np.eye(3), 1.0 * np.eye(3)],
    ])

    # Initial state and covariance
    kalman_filter.x = np.array([*initial_position, 0.0, 0.0, 0.0])
    kalman_filter.P = np.diag([0.01, 0.01, 0.01, 1.0, 1.0, 1.0])

    return kalman_filter

class Localizer:
    def __init__(self):
        load_dotenv()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device:", self.device)

        self.initial_localization_time_seconds = float(os.getenv("INITIAL_LOCALIZATION_TIME_SECONDS", 10.0))
        self.max_walking_speed_per_second = float(os.getenv("MAX_WALKING_SPEED_PER_SECONDS", 2.8))
        self.localization_window_size_seconds = float(os.getenv("LOCALIZATION_WINDOW_SIZE_SECONDS", 2.0))
        self.colmap_to_real_world_scale = float(os.getenv("COLMAP_TO_REAL_WORLD_SCALE", 6.0))
        self.max_distance_between_positions = (self.max_walking_speed_per_second * self.localization_window_size_seconds) / self.colmap_to_real_world_scale
        self.kalman_position_measurement_noise_std = float(os.getenv("KALMAN_POSITION_MEASUREMENT_NOISE_STD", 0.3))
        self.kalman_walking_acceleration_noise_std = float(os.getenv("KALMAN_WALKING_ACCELERATION_NOISE_STD", 0.5))
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
        self.kalman_filter = None

        self.localize_total_time_seconds = 0.0
        self.localize_call_count = 0
        self.localize_last_print_time = None

    def run(self, frame_queue, renderer):
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            renderer.update_video_feed_preview(frame)
            position = self.localize(frame)
            if position is not None:
                renderer.update_target_position(position)

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
            print(f"localize() average time: {average_localize_time_ms:.1f}ms / localize() call")
            self.localize_total_time_seconds = 0.0
            self.localize_call_count = 0
            self.localize_last_print_time = localize_start_time
        return result

    def _localize(self, frame_bgr):
        current_time = time.monotonic()
        top1_index = self.retrieve_top1_index_from_frame(frame_bgr)
        candidate_position = self.index_to_position[top1_index]

        # Initialization phase: collect measurements to seed the filter (get good initial position estimate)
        if self.is_initializing:
            if self.initialization_start_time is None:
                self.initialization_start_time = current_time
            self.last_accepted_time = current_time
            elapsed = current_time - self.initialization_start_time
            if elapsed >= self.initial_localization_time_seconds:
                self.is_initializing = False
                self.kalman_filter = build_pose_kalman_filter(
                    initial_position=candidate_position,
                    position_measurement_noise_std=self.kalman_position_measurement_noise_std,
                    walking_acceleration_noise_std=self.kalman_walking_acceleration_noise_std,
                )
                print("Initial localization complete, Kalman filter initialized")
            return None # No output during init.

        delta_time = current_time - self.last_accepted_time
        if delta_time > self.localization_window_size_seconds:
            print(f"Signal lost... Stay put and look around for {self.initial_localization_time_seconds}s.")
            self.is_initializing = True
            self.initialization_start_time = None
            self.kalman_filter = None
            return None

        # Update state transition matrix with actual elapsed time
        self.kalman_filter.F[:3, 3:] = np.eye(3) * delta_time
        self.kalman_filter.predict()
        predicted_position = self.kalman_filter.x[:3].copy()

        distance_from_prediction = np.linalg.norm(candidate_position - predicted_position)
        if distance_from_prediction > self.max_distance_between_positions:
            print(f"Prediction too far: distance {distance_from_prediction * self.colmap_to_real_world_scale} > max ({self.max_distance_between_positions * self.colmap_to_real_world_scale} meters)")
            return self.kalman_filter.x[:3].copy()

        self.last_accepted_time = current_time
        self.kalman_filter.update(candidate_position)
        return self.kalman_filter.x[:3].copy()
