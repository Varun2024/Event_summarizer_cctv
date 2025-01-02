# # 1st draft

# import cv2
# import os
# import json
# import numpy as np
# import subprocess
# from ultralytics import YOLO
# from tqdm import tqdm
# import torch
# from pytorchvideo.models.hub import slowfast_r50
# from torchvision.transforms import Compose, Lambda
# import av

# # ---------------- CONFIG ---------------- #

# VIDEO_PATH = "input6.mp4"
# OUTPUT_DIR = "outputs"
# CLIP_DIR = os.path.join(OUTPUT_DIR, "clips")

# MOTION_THRESHOLD = 25.0
# CONF_THRESHOLD = 0.4
# PRE_EVENT_SEC = 2.0
# POST_EVENT_SEC = 4.0
# MIN_EVENT_GAP = 1.0

# ACTION_THRESHOLD = 0.6
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# os.makedirs(CLIP_DIR, exist_ok=True)

# # ---------------------------------------- #

# # Load models
# yolo = YOLO("yolov8n.pt")

# action_model = slowfast_r50(pretrained=True)
# action_model = action_model.to(DEVICE).eval()

# # Map Kinetics class indices → readable actions
# KINETICS_ACTION_MAP = {
#     412: "running",
#     405: "walking",
#     401: "falling",
#     386: "punching",
#     387: "kicking",
#     388: "fighting",
#     398: "chasing",
#     390: "interacting"
# }


# def trim_clip(start, end, idx):
#     out_path = f"{CLIP_DIR}/event_{idx:03d}.mp4"
#     cmd = [
#         "ffmpeg", "-y",
#         "-i", VIDEO_PATH,
#         "-ss", str(start),
#         "-to", str(end),
#         "-c:v", "libx264",
#         "-preset", "fast",
#         "-crf", "23",
#         out_path
#     ]
#     subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#     return out_path


# def merge_intervals(intervals):
#     intervals.sort()
#     merged = []
#     for s, e in intervals:
#         if not merged or s > merged[-1][1] + MIN_EVENT_GAP:
#             merged.append([s, e])
#         else:
#             merged[-1][1] = max(merged[-1][1], e)
#     return merged


# def load_clip_for_action(video_path, num_frames=32):
#     container = av.open(video_path)
#     frames = []
#     for frame in container.decode(video=0):
#         img = frame.to_rgb().to_ndarray()
#         img = cv2.resize(img, (224, 224))
#         frames.append(img)
#         if len(frames) >= num_frames:
#             break

#     frames = np.stack(frames) / 255.0
#     frames = torch.tensor(frames).permute(3, 0, 1, 2)  # C T H W
#     return frames.unsqueeze(0).to(DEVICE)


# def run_action_recognition(clip_path):
#     try:
#         clip = load_clip_for_action(clip_path)
#         with torch.no_grad():
#             preds = action_model(clip)
#             probs = torch.softmax(preds, dim=1)[0]

#         for idx, label in KINETICS_ACTION_MAP.items():
#             if probs[idx].item() > ACTION_THRESHOLD:
#                 return label, float(probs[idx].item())

#     except Exception:
#         pass

#     return None, 0.0


# def main():
#     cap = cv2.VideoCapture(VIDEO_PATH)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     duration = frame_count / fps

#     prev_gray = None
#     candidate_times = []

#     print("🔍 Fast-pass scan")

#     for frame_idx in tqdm(range(frame_count)):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         t = frame_idx / fps
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         motion = 0
#         if prev_gray is not None:
#             motion = cv2.absdiff(gray, prev_gray).mean()
#         prev_gray = gray

#         person_flag = False
#         if frame_idx % int(fps) == 0:
#             results = yolo.predict(frame, conf=CONF_THRESHOLD, verbose=False)
#             for r in results:
#                 for box in r.boxes:
#                     if yolo.names[int(box.cls[0])] == "person":
#                         person_flag = True

#         if motion > MOTION_THRESHOLD or person_flag:
#             candidate_times.append((max(0, t - PRE_EVENT_SEC), min(duration, t + POST_EVENT_SEC)))

#     cap.release()
#     candidate_times = merge_intervals(candidate_times)

#     print(f"🎯 Candidates: {len(candidate_times)}")

#     events = []

#     for idx, (start, end) in enumerate(candidate_times):
#         clip_path = trim_clip(start, end, idx)

#         results = yolo.predict(clip_path, conf=CONF_THRESHOLD, verbose=False)

#         persons = 0
#         weapons = []

#         for r in results:
#             for box in r.boxes:
#                 label = yolo.names[int(box.cls[0])]
#                 if label == "person":
#                     persons += 1
#                 if label in ["knife", "gun"]:
#                     weapons.append(label)

#         action, action_conf = run_action_recognition(clip_path)

#         # -------- FUSION -------- #
#         severity = "low"
#         score = 0.3

#         if persons >= 2:
#             score += 0.2
#         if action:
#             score += 0.3
#         if weapons:
#             score += 0.4
#             severity = "high"
#         elif action in ["fighting", "punching", "kicking"]:
#             severity = "high"
#         elif persons >= 2:
#             severity = "medium"

#         events.append({
#             "event_id": f"event_{idx:03d}",
#             "start": round(start, 2),
#             "end": round(end, 2),
#             "persons": persons,
#             "weapons": weapons,
#             "action": action,
#             "action_conf": round(action_conf, 2),
#             "severity": severity,
#             "score": round(score, 2),
#             "clip": clip_path
#         })

#     manifest = {
#         "video": VIDEO_PATH,
#         "duration_sec": round(duration, 2),
#         "events": events
#     }

#     with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
#         json.dump(manifest, f, indent=2)

#     print("✅ Done with action recognition!")


# if __name__ == "__main__":
#     main()





# Draft version for micro-violence detection


import cv2
import os
import json
import numpy as np
import subprocess
from ultralytics import YOLO
from tqdm import tqdm
import torch
from pytorchvideo.models.hub import slowfast_r50
import av

# ================= CONFIG ================= #

VIDEO_PATH = "input.mp4"
OUTPUT_DIR = "outputs"
CLIP_DIR = os.path.join(OUTPUT_DIR, "clips")
MICRO_CLIP_DIR = os.path.join(OUTPUT_DIR, "micro_clips")

os.makedirs(CLIP_DIR, exist_ok=True)
os.makedirs(MICRO_CLIP_DIR, exist_ok=True)

# Motion & activity
MOTION_THRESHOLD = 0.6
ACTIVITY_WINDOW_SEC = 1.0
ACTIVITY_RATIO = 0.4
INACTIVITY_GAP_SEC = 3.0

# Micro-violence
MICRO_SPIKE_MULTIPLIER = 3.0
MICRO_MAX_DURATION = 0.7
MICRO_MERGE_GAP = 1.0

# Clip padding
PRE_EVENT_SEC = 1.5
POST_EVENT_SEC = 2.5

# Presentation chunking
MAX_EVENT_DURATION = 30.0

CONF_THRESHOLD = 0.4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= MODELS ================= #

yolo = YOLO("yolov8n.pt")
action_model = slowfast_r50(pretrained=True).to(DEVICE).eval()

KINETICS_ACTION_MAP = {
    388: "fighting",
    386: "punching",
    387: "kicking",
    390: "interacting",
    412: "running",
    401: "falling",
}

# ================= UTILS ================= #

def ffmpeg_clip(start, end, out_path):
    cmd = ["ffmpeg", "-y", "-i", VIDEO_PATH, "-ss", str(start), "-to", str(end),
           "-c:v", "libx264", "-preset", "fast", "-crf", "23", out_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def merge_times(times, gap):
    merged = []
    for t in sorted(times):
        if not merged or t - merged[-1] > gap:
            merged.append(t)
    return merged

# ================= MAIN ================= #

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps

    prev_gray = None
    motion_vals = []
    micro_spikes = []

    print("🔍 Pass 1: Motion analysis")

    for i in tqdm(range(frames)):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = 0 if prev_gray is None else cv2.absdiff(gray, prev_gray).mean()
        prev_gray = gray
        motion_vals.append(motion)

    cap.release()

    baseline_motion = np.median(motion_vals)

    print(f"📏 Baseline motion: {baseline_motion:.3f}")

    # ================= MICRO EVENT DETECTION ================= #

    cap = cv2.VideoCapture(VIDEO_PATH)

    for i, motion in enumerate(motion_vals):
        t = i / fps

        if motion > baseline_motion * MICRO_SPIKE_MULTIPLIER:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            results = yolo.predict(frame, conf=CONF_THRESHOLD, verbose=False)
            persons = sum(
                1 for r in results for b in r.boxes
                if yolo.names[int(b.cls[0])] == "person"
            )

            if persons >= 2:
                micro_spikes.append(t)

    cap.release()

    micro_spikes = merge_times(micro_spikes, MICRO_MERGE_GAP)

    micro_events = []
    print(f"⚡ Micro events detected: {len(micro_spikes)}")

    for idx, t in enumerate(micro_spikes):
        start = max(0, t - 0.5)
        end = min(duration, t + 0.5)
        out = f"{MICRO_CLIP_DIR}/micro_{idx:03d}.mp4"
        ffmpeg_clip(start, end, out)

        micro_events.append({
            "id": f"micro_{idx:03d}",
            "time": round(t, 2),
            "type": "micro_violence (slap/push)",
            "clip": out
        })

    # ================= MACRO EVENT (SUMMARY) ================= #

    macro_events = []
    if duration > 0:
        ffmpeg_clip(0, duration, f"{CLIP_DIR}/summary.mp4")
        macro_events.append({
            "id": "macro_000",
            "start": 0,
            "end": round(duration, 2),
            "type": "scene_activity_summary",
            "clip": f"{CLIP_DIR}/summary.mp4"
        })

    manifest = {
        "video": VIDEO_PATH,
        "duration": round(duration, 2),
        "macro_events": macro_events,
        "micro_events": micro_events
    }

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("✅ Event summarization complete")

if __name__ == "__main__":
    main()
