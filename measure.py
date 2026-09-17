"""
Webcam object measurement using an A4 sheet of paper as reference.

Pipeline:
 1. The trained YOLOv8 model (best.pt) locates the A4 sheet in the frame.
 2. The sheet's 4 corners are located inside the detection box. If they can't
    be found, the raw detection box is used instead (measurement is then only
    approximate; the camera should look straight down at the sheet).
 3. The sheet is warped into a flat 210 x 297 mm image via a perspective
    transform, so every pixel maps to a fixed mm value regardless of the
    camera angle.
 4. Objects placed ON TOP of the sheet (darker or more saturated than the
    paper) are detected, and their length/width are reported in mm.

Usage:  python measure.py [--model best.pt] [--camera 0] [--conf 0.5]
Keys:   q = quit, s = save a screenshot
"""
import argparse
import time

import cv2
import numpy as np

A4_SHORT_MM, A4_LONG_MM = 210.0, 297.0
PX_PER_MM = 3            # resolution of the flattened paper image
EDGE_MARGIN_MM = 6       # ignore shadows/corner artifacts near the paper edge
MIN_AREA_MM2 = 150       # blobs smaller than this are not counted as objects
DARKNESS_THRESHOLD = 45      # object must be at least this much darker than the paper (0-255)
SATURATION_THRESHOLD = 50    # ...or at least this much more saturated than the paper (0-255)
SMOOTHING = 0.6          # reduces corner jitter (0 = off, closer to 1 = smoother)


def order_corners(pts):
    """Orders 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    total = pts.sum(axis=1)
    diff = pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmin(total)], pts[np.argmin(diff)],
                     pts[np.argmax(total)], pts[np.argmax(diff)]], dtype=np.float32)


def find_paper_corners(frame, box):
    """Looks for the paper's real 4 corners inside the YOLO box. Returns None if not found."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)
    pad_x, pad_y = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
    rx1, ry1 = max(0, int(x1) - pad_x), max(0, int(y1) - pad_y)
    rx2, ry2 = min(w, int(x2) + pad_x), min(h, int(y2) + pad_y)
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None

    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Objects on the paper can stick out over its edge; the convex hull closes that gap
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    area = cv2.contourArea(hull)
    if not (0.5 * box_area < area < 1.3 * box_area):
        return None  # a bright background would also throw the paper detection off
    approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    if len(approx) != 4:
        return None
    return order_corners(approx.reshape(4, 2) + [rx1, ry1])


def compute_transform(corners):
    """Computes the matrix that warps the frame into an mm-scaled flat paper image."""
    top_left, top_right, bottom_right, bottom_left = corners
    width_sum = np.linalg.norm(top_right - top_left) + np.linalg.norm(bottom_right - bottom_left)
    height_sum = np.linalg.norm(bottom_left - top_left) + np.linalg.norm(bottom_right - top_right)
    paper_w_mm, paper_h_mm = (A4_LONG_MM, A4_SHORT_MM) if width_sum >= height_sum else (A4_SHORT_MM, A4_LONG_MM)
    out_w, out_h = int(paper_w_mm * PX_PER_MM), int(paper_h_mm * PX_PER_MM)
    dst_pts = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    return cv2.getPerspectiveTransform(corners, dst_pts), (out_w, out_h)


def measure_objects(flat):
    """Finds objects on the flattened paper: [(minAreaRect, length_mm, width_mm), ...]"""
    gray = cv2.GaussianBlur(cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    saturation = cv2.cvtColor(flat, cv2.COLOR_BGR2HSV)[:, :, 1]
    mask = ((gray < np.median(gray) - DARKNESS_THRESHOLD) |
            (saturation > np.median(saturation) + SATURATION_THRESHOLD)).astype(np.uint8) * 255

    margin_px = EDGE_MARGIN_MM * PX_PER_MM
    mask[:margin_px, :] = 0
    mask[-margin_px:, :] = 0
    mask[:, :margin_px] = 0
    mask[:, -margin_px:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    results = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA_MM2 * PX_PER_MM ** 2:
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        results.append((rect, max(rw, rh) / PX_PER_MM, min(rw, rh) / PX_PER_MM))
    return results


def draw_text(img, text, pos, color=(255, 255, 255), scale=0.55):
    x, y = int(pos[0]), int(pos[1])
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="A4-referenced webcam measurement")
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    from ultralytics import YOLO
    import torch

    model = YOLO(args.model)
    device = 0 if torch.cuda.is_available() else "cpu"
    print("Model classes:", model.names, "| device:", device)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    prev_corners = None
    last_time = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        display = frame.copy()

        result = model(frame, device=device, conf=args.conf, verbose=False)[0]
        if len(result.boxes) == 0:
            prev_corners = None
            draw_text(display, "A4 paper not found", (10, 30), (0, 0, 255), 0.7)
        else:
            best_idx = int(result.boxes.conf.argmax())
            box = result.boxes.xyxy[best_idx].cpu().numpy()
            confidence = float(result.boxes.conf[best_idx])

            corners = find_paper_corners(frame, box)
            corners_found = corners is not None
            if not corners_found:
                x1, y1, x2, y2 = box
                corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

            # Smooth out frame-to-frame jitter; snap immediately if the paper actually moved
            if prev_corners is not None and np.abs(corners - prev_corners).mean() < 15:
                corners = SMOOTHING * prev_corners + (1 - SMOOTHING) * corners
            prev_corners = corners

            M, size = compute_transform(corners)
            flat = cv2.warpPerspective(frame, M, size)
            M_inv = np.linalg.inv(M)

            paper_color = (0, 255, 0) if corners_found else (0, 200, 255)
            cv2.polylines(display, [corners.astype(np.int32)], True, paper_color, 2)
            status = "corners detected" if corners_found else "APPROX (corners not found)"
            draw_text(display, f"A4 {confidence * 100:.0f}% - {status}", (10, 30), paper_color, 0.6)

            for rect, length, width in measure_objects(flat):
                box_flat = cv2.boxPoints(rect).reshape(-1, 1, 2).astype(np.float32)
                box_frame = cv2.perspectiveTransform(box_flat, M_inv).reshape(-1, 2)
                cv2.polylines(display, [box_frame.astype(np.int32)], True, (255, 0, 255), 2)
                cx, cy = box_frame.mean(axis=0)
                draw_text(display, f"{length / 10:.1f} x {width / 10:.1f} cm", (cx - 50, cy))

                cv2.drawContours(flat, [cv2.boxPoints(rect).astype(np.int32)], 0, (255, 0, 255), 2)
                draw_text(flat, f"{length:.0f} x {width:.0f} mm", rect[0], scale=0.6)

            cv2.imshow("Flattened A4", flat)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last_time, 1e-6)
        last_time = now
        draw_text(display, f"FPS {fps:.0f}", (10, display.shape[0] - 10))
        cv2.imshow("Measurement", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            filename = time.strftime("capture_%Y%m%d_%H%M%S.png")
            cv2.imwrite(filename, display)
            print("Saved:", filename)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
