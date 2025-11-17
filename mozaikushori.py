# mozaikushori.py
import os
import cv2
import torch
import numpy as np
from PIL import Image
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords
import random

# ---------------------------
# モデルロード関数
# ---------------------------
def load_model_s(weights_s, device):
    return attempt_load(weights_s, map_location=device)

def load_model_m(weights_m, device):
    return attempt_load(weights_m, map_location=device)

def load_model_l(weights_l, device):
    return attempt_load(weights_l, map_location=device)

# ---------------------------
# モザイク処理
# ---------------------------
def apply_mosaic(img, xyxy, mosaic_size=10):
    x1, y1, x2, y2 = map(int, xyxy)
    face = img[y1:y2, x1:x2]
    if face.size > 0:
        h, w = face.shape[:2]
        small = cv2.resize(face, (mosaic_size, mosaic_size), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        img[y1:y2, x1:x2] = mosaic
    return img

# ---------------------------
# IOU計算
# ---------------------------
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_ - x1_) * (y2_ - y1_)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0

# ---------------------------
# 顔領域重複統合
# ---------------------------
def merge_overlapping_faces(detections, iou_thresh=0.4):
    merged = []
    for det in detections:
        x1, y1, x2, y2 = det
        merged_flag = False
        for i, existing in enumerate(merged):
            if iou(existing, (x1, y1, x2, y2)) > iou_thresh:
                ex_x1, ex_y1, ex_x2, ex_y2 = existing
                merged[i] = (
                    int((ex_x1 + x1) / 2),
                    int((ex_y1 + y1) / 2),
                    int((ex_x2 + x2) / 2),
                    int((ex_y2 + y2) / 2)
                )
                merged_flag = True
                break
        if not merged_flag:
            merged.append((x1, y1, x2, y2))
    return merged

# ---------------------------
# 結果描画（モザイク or 絵文字）
# ---------------------------
def show_results(img, xyxy, s_img, mode="mosaic", emoji_path=None, placed_boxes=None, emoji_scale=1.8):
    x1, y1, x2, y2 = map(int, xyxy)
    h, w, _ = img.shape
    face_area_ratio = ((x2 - x1) * (y2 - y1)) / (h * w)
    s_img.append(face_area_ratio)

    # 他の絵文字と重なりすぎる場合はスキップ
    if placed_boxes:
        for pb in placed_boxes:
            if iou(pb, (x1, y1, x2, y2)) > 0.3:
                return img, False

    if mode == "mosaic":
        img = apply_mosaic(img, [x1, y1, x2, y2], mosaic_size=10)
    elif mode == "emoji" and emoji_path:
        try:
            emoji_img = Image.open(emoji_path).convert("RGBA")

            scale = emoji_scale
            w = int((x2 - x1) * scale)
            h = int((y2 - y1) * scale)

            emoji_resized = emoji_img.resize((w, h), Image.LANCZOS)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            x_start = cx - w // 2
            y_start = cy - h // 2

            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
            img_pil.paste(emoji_resized, (x_start, y_start), emoji_resized)
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGR)
        except Exception as e:
            print("⚠️ 絵文字貼り付けでエラー:", e)
            return img, False

    return img, True

# ---------------------------
# 顔検出＆処理（順番モード対応版）
# ---------------------------
def detect_one(im, model_s, model_m, model_l, device,
               emoji_path=None, save_path=None,
               mode="emoji", emoji_folder=None,
               emoji_scale=1.8, sequential_mode=False):
    img0 = im.copy() if isinstance(im, np.ndarray) else np.array(im)

    # 前処理
    img = letterbox(img0, new_shape=640)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # 推論
    pred_s = model_s(img)[0]
    pred_m = model_m(img)[0]
    pred_l = model_l(img)[0]

    # 非極大抑制
    det_s = non_max_suppression_face(pred_s, conf_thres=0.3, iou_thres=0.5)[0]
    det_m = non_max_suppression_face(pred_m, conf_thres=0.3, iou_thres=0.5)[0]
    det_l = non_max_suppression_face(pred_l, conf_thres=0.3, iou_thres=0.5)[0]

    detections = []
    for det in [det_s, det_m, det_l]:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for j in range(det.size(0)):
                x1, y1, x2, y2 = map(int, det[j, :4])
                detections.append((x1, y1, x2, y2))

    # 顔の統合
    merged_faces = merge_overlapping_faces(detections, iou_thresh=0.4)

    # 🟢 順番モードのときは左から順に処理
    if sequential_mode:
        merged_faces = sorted(merged_faces, key=lambda box: box[0])

    s_img = []
    placed_boxes = []

    # 🟢 絵文字リストを順番に使うための初期化
    emoji_files = []
    emoji_index = 0
    if emoji_folder:
        all_subfolders = [os.path.join(emoji_folder, f) for f in os.listdir(emoji_folder)
                          if os.path.isdir(os.path.join(emoji_folder, f))]
        chosen_folder = random.choice(all_subfolders) if all_subfolders else emoji_folder
        emoji_files = [os.path.join(chosen_folder, f) for f in os.listdir(chosen_folder)
                       if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        emoji_files.sort()

    # 🟢 顔ごとに処理
    for (x1, y1, x2, y2) in merged_faces:
        selected_emoji_path = emoji_path

        if mode == "emoji" and emoji_folder and emoji_files:
            if sequential_mode:
                selected_emoji_path = emoji_files[emoji_index % len(emoji_files)]
                emoji_index += 1
            else:
                selected_emoji_path = random.choice(emoji_files)

        img0, pasted = show_results(
            img0, (x1, y1, x2, y2), s_img,
            mode=mode, emoji_path=selected_emoji_path,
            placed_boxes=placed_boxes,
            emoji_scale=emoji_scale
        )
        if pasted:
            placed_boxes.append((x1, y1, x2, y2))

    if save_path:
        cv2.imwrite(save_path, img0)

    return img0
