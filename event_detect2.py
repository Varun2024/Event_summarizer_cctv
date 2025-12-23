

# import cv2
# import os
# import json
# import numpy as np
# import subprocess
# import torch
# import av
# from ultralytics import YOLO
# from tqdm import tqdm
# from pytorchvideo.models.hub import slowfast_r50
# import supervision as sv

# # ================= CONFIG ================= #

# VIDEO_PATH = "input6.mp4"
# OUTPUT_DIR = "outputs"
# EVENT_CLIP_DIR = os.path.join(OUTPUT_DIR, "events")

# os.makedirs(EVENT_CLIP_DIR, exist_ok=True)

# CONF_THRESHOLD = 0.4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Interaction thresholds
# VELOCITY_SPIKE = 20.0              # pixel distance jump
# MIN_INTERACTION_FRAMES = 6         # sustain interaction
# EVENT_PADDING = 0.7                # seconds before/after event

# # ================= MODELS ================= #

# yolo = YOLO("yolov8n.pt")
# tracker = sv.ByteTrack()

# action_model = slowfast_r50(pretrained=True).to(DEVICE).eval()

# KINETICS_ACTION_MAP = {
#     388: "fighting",
#     386: "punching",
#     387: "kicking",
#     401: "falling",
#     412: "running",
#     398: "chasing",
#     390: "interacting",
#     404: "shoving",
#     405: "walking",
#     406: "standing",
#     407: "sitting",
#     409: "pushing",
#     410: "grappling",
#     411: "wrestling",
# }
# # ================= UTILS ================= #

# def ffmpeg_clip(start, end, out):
#     subprocess.run(
#         ["ffmpeg", "-y", "-i", VIDEO_PATH, "-ss", str(start), "-to", str(end),
#          "-c:v", "libx264", "-crf", "23", out],
#         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
#     )

# def center(box):
#     x1, y1, x2, y2 = box
#     return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

# def slowfast_input(clip_path, frames=32):
#     container = av.open(clip_path)
#     imgs = []

#     for frame in container.decode(video=0):
#         img = cv2.resize(frame.to_rgb().to_ndarray(), (224, 224))
#         imgs.append(img)
#         if len(imgs) >= frames:
#             break

#     if len(imgs) < 8:
#         return None

#     imgs += [imgs[-1]] * (frames - len(imgs))
#     imgs = torch.tensor(imgs).permute(3, 0, 1, 2).float() / 255.0

#     fast = imgs.unsqueeze(0).to(DEVICE)
#     slow = imgs[:, ::4].unsqueeze(0).to(DEVICE)
#     return [slow, fast]

# def classify_action(clip_path):
#     inp = slowfast_input(clip_path)
#     if inp is None:
#         return ["physical_interaction"]

#     with torch.no_grad():
#         probs = torch.softmax(action_model(inp), dim=1)[0]

#     labels = []
#     topk = torch.topk(probs, 5)

#     for idx, conf in zip(topk.indices, topk.values):
#         idx = int(idx.item())
#         if idx in KINETICS_ACTION_MAP and conf.item() > 0.15:
#             labels.append(KINETICS_ACTION_MAP[idx])

#     return list(set(labels)) if labels else ["physical_interaction"]

# # ================= MAIN ================= #

# def main():
#     cap = cv2.VideoCapture(VIDEO_PATH)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     active_interactions = {}
#     detected_times = []

#     print("🔍 Detecting significant interactions")

#     for frame_idx in tqdm(range(total_frames)):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         t = frame_idx / fps

#         results = yolo(frame, conf=CONF_THRESHOLD, verbose=False)[0]
#         detections = sv.Detections.from_ultralytics(results)
#         detections = tracker.update_with_detections(detections)

#         people = detections[detections.class_id == 0]

#         for i in range(len(people)):
#             for j in range(i + 1, len(people)):
#                 p1, p2 = people[i], people[j]

#                 id1 = int(p1.tracker_id)
#                 id2 = int(p2.tracker_id)
#                 key = tuple(sorted((id1, id2)))
#                 p1_flat = p1.xyxy.flatten()
#                 p2_flat = p2.xyxy.flatten()
#                 c1 = center(p1_flat)
#                 c2 = center(p2_flat)
#                 dist = np.linalg.norm(c1 - c2)

#                 if key not in active_interactions:
#                     active_interactions[key] = {
#                         "start": t,
#                         "last_dist": dist,
#                         "frames": 1
#                     }
#                 else:
#                     prev = active_interactions[key]
#                     speed = abs(prev["last_dist"] - dist)

#                     prev["frames"] += 1
#                     prev["last_dist"] = dist

#                     if speed > VELOCITY_SPIKE and prev["frames"] >= MIN_INTERACTION_FRAMES:
#                         detected_times.append(t)
#                         active_interactions.pop(key, None)

#     cap.release()

#     print(f"⚡ Significant events detected: {len(detected_times)}")

#     events = []
#     for idx, t in enumerate(sorted(set(detected_times))):
#         start = max(0, t - EVENT_PADDING)
#         end = t + EVENT_PADDING

#         clip_path = os.path.join(EVENT_CLIP_DIR, f"event_{idx:03d}.mp4")
#         ffmpeg_clip(start, end, clip_path)

#         labels = classify_action(clip_path)

#         severity = "low"
#         if any(x in labels for x in ["fighting", "punching", "kicking"]):
#             severity = "high"
#         elif "falling" in labels:
#             severity = "medium"

#         events.append({
#             "id": f"event_{idx:03d}",
#             "time": round(t, 2),
#             "labels": labels,
#             "severity": severity,
#             "clip": clip_path
#         })

#     manifest = {
#         "video": VIDEO_PATH,
#         "events": events
#     }

#     with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
#         json.dump(manifest, f, indent=2)

#     print("✅ Significant event extraction complete")

# if __name__ == "__main__":
#     main()


import cv2
import os
import json
import numpy as np
import subprocess
import torch
import av
from ultralytics import YOLO
from tqdm import tqdm
from pytorchvideo.models.hub import slowfast_r50
import supervision as sv

# ================= CONFIG ================= #

VIDEO_PATH = "input.mp4"
OUTPUT_DIR = "outputs"
EVENT_CLIP_DIR = os.path.join(OUTPUT_DIR, "events")

os.makedirs(EVENT_CLIP_DIR, exist_ok=True)

CONF_THRESHOLD = 0.4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Interaction thresholds
VELOCITY_SPIKE = 25.0
MIN_INTERACTION_FRAMES = 6
EVENT_PADDING = 0.7  # seconds

# ================= MODELS ================= #

yolo = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()

action_model = slowfast_r50(pretrained=True).to(DEVICE).eval()

KINETICS_ACTION_MAP = {
    388: "fighting",
    386: "punching",
    387: "kicking",
    401: "falling",
    398: "chasing",
    412: "running",
    409: "pushing",
    410: "grappling",
    390: "interacting"
}

# ================= UTILS ================= #

def ffmpeg_clip(start, end, out):
    subprocess.run(
        ["ffmpeg", "-y", "-i", VIDEO_PATH,
         "-ss", str(start), "-to", str(end),
         "-c:v", "libx264", "-crf", "23", out],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def center_xyxy(box):
    """
    box can be shape (4,) or (1,4)
    """
    box = np.array(box).reshape(-1)
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def slowfast_input(clip_path, frames=32):
    container = av.open(clip_path)
    imgs = []

    for frame in container.decode(video=0):
        img = cv2.resize(frame.to_rgb().to_ndarray(), (224, 224))
        imgs.append(img)
        if len(imgs) >= frames:
            break

    if len(imgs) < 8:
        return None

    imgs = np.array(imgs, dtype=np.float32) / 255.0
    if imgs.shape[0] < frames:
        imgs = np.concatenate(
            [imgs, np.repeat(imgs[-1][None], frames - imgs.shape[0], axis=0)],
            axis=0
        )

    imgs = torch.from_numpy(imgs).permute(3, 0, 1, 2)
    fast = imgs.unsqueeze(0).to(DEVICE)
    slow = imgs[:, ::4].unsqueeze(0).to(DEVICE)
    return [slow, fast]


def classify_action(clip_path):
    inp = slowfast_input(clip_path)
    if inp is None:
        return []

    with torch.no_grad():
        probs = torch.softmax(action_model(inp), dim=1)[0]

    labels = []
    for idx, conf in enumerate(probs):
        if idx in KINETICS_ACTION_MAP and conf > 0.2:
            labels.append(KINETICS_ACTION_MAP[idx])

    return list(set(labels))


# ================= MAIN ================= #

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    active_interactions = {}
    detected_times = []

    print("🔍 Detecting significant interactions")

    for frame_idx in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps

        results = yolo(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        people = detections[detections.class_id == 0]

        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                p1, p2 = people[i], people[j]

                if p1.tracker_id is None or p2.tracker_id is None:
                    continue

                id1 = int(p1.tracker_id)
                id2 = int(p2.tracker_id)
                key = tuple(sorted((id1, id2)))

                c1 = center_xyxy(p1.xyxy)
                c2 = center_xyxy(p2.xyxy)
                dist = np.linalg.norm(c1 - c2)

                if key not in active_interactions:
                    active_interactions[key] = {
                        "last_dist": dist,
                        "frames": 1
                    }
                else:
                    prev = active_interactions[key]
                    speed = abs(prev["last_dist"] - dist)
                    prev["frames"] += 1
                    prev["last_dist"] = dist

                    if speed > VELOCITY_SPIKE and prev["frames"] >= MIN_INTERACTION_FRAMES:
                        detected_times.append(t)
                        active_interactions.pop(key, None)

    cap.release()

    detected_times = sorted(set(detected_times))
    print(f"⚡ Significant events detected: {len(detected_times)}")

    events = []

    for idx, t in enumerate(detected_times):
        start = max(0, t - EVENT_PADDING)
        end = t + EVENT_PADDING

        clip_path = os.path.join(EVENT_CLIP_DIR, f"event_{idx:03d}.mp4")
        ffmpeg_clip(start, end, clip_path)

        labels = classify_action(clip_path)

        severity = "low"
        if any(x in labels for x in ["fighting", "punching", "kicking", "grappling"]):
            severity = "high"
        elif "falling" in labels:
            severity = "medium"

        events.append({
            "id": f"event_{idx:03d}",
            "time": round(t, 2),
            "labels": labels,
            "severity": severity,
            "clip": clip_path
        })

    manifest = {
        "video": VIDEO_PATH,
        "events": events
    }

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("✅ Significant event extraction complete")


if __name__ == "__main__":
    main()
