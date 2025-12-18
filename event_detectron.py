import cv2
import os
import json
import numpy as np
import subprocess
from ultralytics import YOLO
from tqdm import tqdm
import torch
from pytorchvideo.models.hub import slowfast_r50
from torchvision.transforms import Compose, Lambda
import av

# ---------------- CONFIG ---------------- #

VIDEO_PATH = "input.mp4"
OUTPUT_DIR = "outputs"
CLIP_DIR = os.path.join(OUTPUT_DIR, "clips")

MOTION_THRESHOLD = 25.0
CONF_THRESHOLD = 0.4
PRE_EVENT_SEC = 2.0
POST_EVENT_SEC = 4.0
MIN_EVENT_GAP = 1.0

ACTION_THRESHOLD = 0.6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CLIP_DIR, exist_ok=True)

# ---------------------------------------- #

# Load models
yolo = YOLO("yolov8n.pt")

action_model = slowfast_r50(pretrained=True)
action_model = action_model.to(DEVICE).eval()

# Kinetics labels (simplified)
KINETICS_CLASSES = {
    386: "punching",
    387: "kicking",
    388: "fighting",
    412: "running",
    401: "falling"
}


def trim_clip(start, end, idx):
    out_path = f"{CLIP_DIR}/event_{idx:03d}.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", VIDEO_PATH,
        "-ss", str(start),
        "-to", str(end),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def merge_intervals(intervals):
    intervals.sort()
    merged = []
    for s, e in intervals:
        if not merged or s > merged[-1][1] + MIN_EVENT_GAP:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return merged


def load_clip_for_action(video_path, num_frames=32):
    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_rgb().to_ndarray()
        img = cv2.resize(img, (224, 224))
        frames.append(img)
        if len(frames) >= num_frames:
            break

    frames = np.stack(frames) / 255.0
    frames = torch.tensor(frames).permute(3, 0, 1, 2)  # C T H W
    return frames.unsqueeze(0).to(DEVICE)


def run_action_recognition(clip_path):
    try:
        clip = load_clip_for_action(clip_path)
        with torch.no_grad():
            preds = action_model(clip)
            probs = torch.softmax(preds, dim=1)[0]

        for idx, label in KINETICS_CLASSES.items():
            if probs[idx].item() > ACTION_THRESHOLD:
                return label, float(probs[idx].item())

    except Exception:
        pass

    return None, 0.0


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    prev_gray = None
    candidate_times = []

    print("🔍 Fast-pass scan")

    for frame_idx in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        motion = 0
        if prev_gray is not None:
            motion = cv2.absdiff(gray, prev_gray).mean()
        prev_gray = gray

        person_flag = False
        if frame_idx % int(fps) == 0:
            results = yolo.predict(frame, conf=CONF_THRESHOLD, verbose=False)
            for r in results:
                for box in r.boxes:
                    if yolo.names[int(box.cls[0])] == "person":
                        person_flag = True

        if motion > MOTION_THRESHOLD or person_flag:
            candidate_times.append((max(0, t - PRE_EVENT_SEC), min(duration, t + POST_EVENT_SEC)))

    cap.release()
    candidate_times = merge_intervals(candidate_times)

    print(f"🎯 Candidates: {len(candidate_times)}")

    events = []

    for idx, (start, end) in enumerate(candidate_times):
        clip_path = trim_clip(start, end, idx)

        results = yolo.predict(clip_path, conf=CONF_THRESHOLD, verbose=False)

        persons = 0
        weapons = []

        for r in results:
            for box in r.boxes:
                label = yolo.names[int(box.cls[0])]
                if label == "person":
                    persons += 1
                if label in ["knife", "gun"]:
                    weapons.append(label)

        action, action_conf = run_action_recognition(clip_path)

        # -------- FUSION -------- #
        severity = "low"
        score = 0.3

        if persons >= 2:
            score += 0.2
        if action:
            score += 0.3
        if weapons:
            score += 0.4
            severity = "high"
        elif action in ["fighting", "punching", "kicking"]:
            severity = "high"
        elif persons >= 2:
            severity = "medium"

        events.append({
            "event_id": f"event_{idx:03d}",
            "start": round(start, 2),
            "end": round(end, 2),
            "persons": persons,
            "weapons": weapons,
            "action": action,
            "action_conf": round(action_conf, 2),
            "severity": severity,
            "score": round(score, 2),
            "clip": clip_path
        })

    manifest = {
        "video": VIDEO_PATH,
        "duration_sec": round(duration, 2),
        "events": events
    }

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("✅ Done with action recognition!")


if __name__ == "__main__":
    main()
