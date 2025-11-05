import cv2
import numpy as np

# ---------------- Constants ---------------- #
VIDEO_PATH = "IMG_3055.mov"
OUTPUT_OVERLAY_PATH = "output_overlay.mp4"   # Overlayed video
OUTPUT_MASK_PATH = "output_mask.mp4"         # Mask-only video

LOWER_PURPLE = np.array([135, 60, 80])
UPPER_PURPLE = np.array([175, 255, 255])

LOWER_GREEN = np.array([38, 55, 70])
UPPER_GREEN = np.array([85, 255, 255])

BALL_DIAMETER_CM = {
    "purple": 12.7,
    "green": 12.7,
}

FOCAL_LENGTH_PX = 800 

MORPH_KERNEL_SIZE = 3
GAUSSIAN_BLUR = (3, 3)
CIRCLE_MIN_DIST = 20
CIRCLE_DP = 1.2
CIRCLE_PARAM1 = 50
CIRCLE_PARAM2 = 16
CIRCLE_MIN_RADIUS = 5
CIRCLE_MAX_RADIUS = 100
COVERAGE_THRESHOLD = 0.7
SCALE = 0.6


# ---------------- Functions ---------------- #
def adaptive_kernel(image, base_size=3):
    h, w = image.shape[:2]
    k = max(3, min(base_size * (w // 640), 7))
    return np.ones((k, k), np.uint8)


def remove_nested_circles(circles, containment_tol=1.0):
    if not circles:
        return circles
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    keep = []
    for i, (x1, y1, r1) in enumerate(circles):
        contained = False
        for (x2, y2, r2) in keep:
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist + r1 * containment_tol <= r2:
                contained = True
                break
        if not contained:
            keep.append((x1, y1, r1))
    return keep


def detect_balls(image, COLOR_LOW, COLOR_HIGH):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COLOR_LOW, COLOR_HIGH)
    kernel = adaptive_kernel(image)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, GAUSSIAN_BLUR, 0)

    circles = cv2.HoughCircles(
        mask, cv2.HOUGH_GRADIENT,
        dp=CIRCLE_DP, minDist=CIRCLE_MIN_DIST,
        param1=CIRCLE_PARAM1, param2=CIRCLE_PARAM2,
        minRadius=CIRCLE_MIN_RADIUS, maxRadius=CIRCLE_MAX_RADIUS
    )

    output = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0, :]:
            circle_mask = np.zeros_like(mask)
            cv2.circle(circle_mask, (x, y), r, 255, -1)
            coverage = cv2.countNonZero(cv2.bitwise_and(mask, circle_mask)) / (np.pi * r * r)
            if coverage > COVERAGE_THRESHOLD:
                output.append((x, y, r))
    output = remove_nested_circles(output, 0.1)
    return mask, output


def estimate_depth(focal_length_px, real_diameter_cm, pixel_radius):
    diameter_px = pixel_radius * 2
    if diameter_px == 0:
        return None
    depth_cm = (focal_length_px * real_diameter_cm) / diameter_px
    return depth_cm


# ---------------- Video Processing ---------------- #
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'H264')
out_overlay = cv2.VideoWriter(OUTPUT_OVERLAY_PATH, fourcc, fps, (frame_width, frame_height))
out_mask = cv2.VideoWriter(OUTPUT_MASK_PATH, fourcc, fps, (frame_width, frame_height), isColor=False)

prev_shape = None
kernel = None
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_small = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    if frame_small.shape != prev_shape:
        kernel = adaptive_kernel(frame_small)
        prev_shape = frame_small.shape

    # Detect both colors
    mask_purple, purple_circles = detect_balls(frame_small, LOWER_PURPLE, UPPER_PURPLE)
    mask_green, green_circles = detect_balls(frame_small, LOWER_GREEN, UPPER_GREEN)

    # Combine masks
    combined_mask = cv2.bitwise_or(mask_purple, mask_green)

    # Draw detections and depth
    overlay = frame.copy()
    for x, y, r in purple_circles:
        x, y, r = int(x / SCALE), int(y / SCALE), int(r / SCALE)
        depth = estimate_depth(FOCAL_LENGTH_PX, BALL_DIAMETER_CM["purple"], r)
        cv2.circle(overlay, (x, y), r, (255, 0, 255), 2)
        if depth:
            cv2.putText(overlay, f"{depth:.1f}cm", (x - 30, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    for x, y, r in green_circles:
        x, y, r = int(x / SCALE), int(y / SCALE), int(r / SCALE)
        depth = estimate_depth(FOCAL_LENGTH_PX, BALL_DIAMETER_CM["green"], r)
        cv2.circle(overlay, (x, y), r, (0, 255, 0), 2)
        if depth:
            cv2.putText(overlay, f"{depth:.1f}cm", (x - 30, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Combine mask for visualization
    combined_mask_full = cv2.resize(combined_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
    combined_mask_bgr = cv2.cvtColor(combined_mask_full, cv2.COLOR_GRAY2BGR)
    output_frame = cv2.addWeighted(overlay, 0.8, combined_mask_bgr, 0.2, 0)

    # Write frames
    out_overlay.write(output_frame)
    out_mask.write(combined_mask_full)

    # Show preview
    if frame_idx % 5 == 0:
        cv2.imshow("Detection + Depth", output_frame)
        cv2.imshow("Mask", combined_mask_full)
        if cv2.waitKey(1) == ord('q'):
            break

    frame_idx += 1

# Cleanup
cap.release()
out_overlay.release()
out_mask.release()
cv2.destroyAllWindows()

print(f"Processed overlay video saved to: {OUTPUT_OVERLAY_PATH}")
print(f"Processed mask video saved to: {OUTPUT_MASK_PATH}")
