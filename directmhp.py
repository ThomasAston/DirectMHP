# directmhp.py

import os
import torch
import cv2
import yaml
import numpy as np
from tqdm import tqdm

from .utils.torch_utils import select_device, time_sync
from .utils.general import check_img_size, scale_coords, non_max_suppression
from .utils.datasets import LoadImages
from .utils.plots import plot_3axis_Zaxis
from .models.experimental import attempt_load

class DirectMHPWrapper:
    def __init__(self,
                 weights_path,
                 data_config='data/agora_coco.yaml',
                 imgsz=1280,
                 conf_thres=0.7,
                 iou_thres=0.45,
                 device='',
                 scales=[1],
                 alpha=0.4):

        self.weights_path = weights_path
        self.device = select_device(device)
        self.imgsz = check_img_size(imgsz, s=32)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.scales = scales
        self.alpha = alpha

        # --- Make the path to data_config dynamic ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # If data_config is not already an absolute path, join it with the current_dir
        if not os.path.isabs(data_config):
            data_config = os.path.join(current_dir, data_config)
        
        # Load the data config
        with open(data_config, 'r') as f:
            self.data_cfg = yaml.safe_load(f)

        # Load model weights
        self.model = attempt_load(weights_path, map_location=self.device)
        self.stride = int(self.model.stride.max())
        if self.device.type != 'cpu':
            dummy = torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device)
            self.model(dummy)
        self.model.eval()

    def run_on_video(self, video_path, start=0, end=-1, save_video=False):
        """
        Run DirectMHP inference on a video file.

        :param video_path: Path to the input video file.
        :param start: Start time in seconds.
        :param end: End time in seconds (-1 for remainder of the video).
        :param save_video: If True, saves an annotated video with bounding boxes and angles.
        :return: (predictions, fps) where
                 predictions = list of dicts, each with {'yaw', 'pitch', 'roll'} for each frame
                 fps = frames per second of the original video
        """
        dataset = LoadImages(video_path, img_size=self.imgsz, stride=self.stride, auto=True)
        cap = dataset.cap
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if end == -1:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) - fps * start)
        else:
            n = int(fps * (end - start))

        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Optionally, set up a VideoWriter if we want to save the output
        writer = None
        if save_video:
            out_path = os.path.splitext(video_path)[0] + "_DirectMHP.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        predictions = []
        dataset = tqdm(dataset, desc='Running DirectMHP inference', total=n)
        for i, (path, img, im0, _) in enumerate(dataset):
            # Convert numpy image to tensor
            img_tensor = torch.from_numpy(img).to(self.device).float() / 255.0
            if img_tensor.ndimension() == 3:
                img_tensor = img_tensor.unsqueeze(0)

            out_ori = self.model(img_tensor, augment=True, scales=self.scales)[0]
            out = non_max_suppression(out_ori,
                                      self.conf_thres,
                                      self.iou_thres,
                                      num_angles=self.data_cfg['num_angles'])
            if len(out[0]) == 0:
                # No detections for this frame
                predictions.append({'yaw': float('nan'),
                                    'pitch': float('nan'),
                                    'roll': float('nan')})
            else:
                # For simplicity, use the first detection
                det = out[0][0].cpu().numpy()
                # det format: [x1, y1, x2, y2, conf, class, pitch, yaw, roll]
                pitch = (det[6] - 0.5) * 180
                yaw = (det[7] - 0.5) * 360
                roll = (det[8] - 0.5) * 180
                predictions.append({'yaw': yaw, 'pitch': pitch, 'roll': roll})

                # (Optional) Overlay bounding box and angles on the frame
                x1, y1, x2, y2 = det[:4].astype(int)
                im0_copy = im0.copy()
                cv2.rectangle(im0_copy, (x1, y1), (x2, y2), (0, 255, 255), 2)
                im0_copy = plot_3axis_Zaxis(im0_copy,
                                            yaw, pitch, roll,
                                            tdx=(x1 + x2) / 2,
                                            tdy=(y1 + y2) / 2,
                                            size=max(y2 - y1, x2 - x1) * 0.8,
                                            thickness=2)
                # Blend original and annotated frame
                im0 = cv2.addWeighted(im0, self.alpha, im0_copy, 1 - self.alpha, gamma=0)

            if save_video and writer is not None:
                writer.write(im0)

            if i == n - 1:
                break

        cap.release()
        if writer is not None:
            writer.release()
        return predictions, fps