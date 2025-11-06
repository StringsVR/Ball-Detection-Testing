import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from .types.ColorRange import ColorRange

class RangeFinder:
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters

    def find_color_clusters(self, image_hsv, mask=None):
        if mask is not None:
            image_hsv = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)

        pixels = image_hsv.reshape(-1, 3)
        pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]  # remove black pixels
        if len(pixels) < self.n_clusters:
            return []

        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, batch_size=1000)
        kmeans.fit(pixels)
        centers = np.uint8(kmeans.cluster_centers_)
        return centers

    def compute_color_range(self, image, color_hint):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if color_hint.lower() == "purple":
            mask = cv2.inRange(hsv, (120, 30, 30), (160, 255, 255))
        elif color_hint.lower() == "green":
            mask = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
        else:
            raise ValueError("Unsupported color hint. Use 'purple' or 'green'.")

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        centers = self.find_color_clusters(hsv, mask)
        if len(centers) == 0:
            print(f"No {color_hint} pixels detected.")
            return None

        lower = np.min(centers, axis=0)
        upper = np.max(centers, axis=0)

        lower = np.clip(lower - np.array([6, 31, 31]), 0, 255)
        upper = np.clip(upper + np.array([17, 42, 42]), 0, 255)

        return ColorRange(lower, upper, name=color_hint)

    def find_color_ranges(self, image):
        ranges = {}
        for color_hint in ["purple", "green"]:
            cr = self.compute_color_range(image, color_hint)
            if cr is not None:
                ranges[color_hint] = cr
        return ranges
