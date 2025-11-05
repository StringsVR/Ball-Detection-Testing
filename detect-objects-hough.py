import cv2
import numpy as np
import time

# ---------------- Constants ---------------- #
IMAGE_PATH = "IMG_3466.webp"

# HSV ranges for purple and green
LOWER_PURPLE = np.array([115, 35, 31])
UPPER_PURPLE = np.array([200, 255, 255])
LOWER_GREEN = np.array([32, 48, 44])
UPPER_GREEN = np.array([100, 255, 255])

BALL_DIAMETER_CM = {
    "purple": 12.7,
    "green": 12.7,
}

FOCAL_LENGTH_PX = 320

MORPH_KERNEL_SIZE = 3
GAUSSIAN_BLUR = (3, 3)
CIRCLE_MIN_DIST = 20
CIRCLE_DP = 1.2
CIRCLE_PARAM1 = 55
CIRCLE_PARAM2 = 16
CIRCLE_MIN_RADIUS = 5
CIRCLE_MAX_RADIUS = 100
COVERAGE_THRESHOLD = 0.8


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
    for (x1, y1, r1) in circles:
        contained = False
        for (x2, y2, r2) in keep:
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
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
        mask,
        cv2.HOUGH_GRADIENT,
        dp=CIRCLE_DP,
        minDist=CIRCLE_MIN_DIST,
        param1=CIRCLE_PARAM1,
        param2=CIRCLE_PARAM2,
        minRadius=CIRCLE_MIN_RADIUS,
        maxRadius=CIRCLE_MAX_RADIUS,
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

    output = remove_nested_circles(output, 0.05)
    return mask, output


def estimate_depth(focal_length_px, real_diameter_cm, pixel_radius):
    diameter_px = pixel_radius * 2
    if diameter_px == 0:
        return None
    depth_cm = (focal_length_px * real_diameter_cm) / diameter_px
    return depth_cm


# ---------------- Main ---------------- #
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

overlay = image.copy()

# --- Benchmarking loop ---
N_RUNS = 20
times = []

for _ in range(N_RUNS):
    start = time.perf_counter()

    mask_purple, purple_circles = detect_balls(image, LOWER_PURPLE, UPPER_PURPLE)
    mask_green, green_circles = detect_balls(image, LOWER_GREEN, UPPER_GREEN)

    end = time.perf_counter()
    times.append((end - start) * 1000)  # convert to ms

avg_time = sum(times) / len(times)
print(f"Average pipeline time over {N_RUNS} runs: {avg_time:.2f} ms")

# --- Draw detections ---
for x, y, r in purple_circles:
    depth = estimate_depth(FOCAL_LENGTH_PX, BALL_DIAMETER_CM["purple"], r)
    cv2.circle(overlay, (x, y), r, (255, 0, 255), 2)
    cv2.putText(overlay, f"{depth:.1f}cm", (x - 30, y - r - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

for x, y, r in green_circles:
    depth = estimate_depth(FOCAL_LENGTH_PX, BALL_DIAMETER_CM["green"], r)
    cv2.circle(overlay, (x, y), r, (0, 255, 0), 2)
    cv2.putText(overlay, f"{depth:.1f}cm", (x - 30, y - r - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

combined_mask = cv2.bitwise_or(mask_purple, mask_green)

cv2.imshow("Combined Mask", combined_mask)
cv2.imshow("Detected Balls + Depth", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
