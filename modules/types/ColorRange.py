import numpy as np
import json
import cv2

class ColorRange:
    def __init__(self, lower_color, upper_color, name = None):
        self.lower = np.array(lower_color, dtype=np.uint8)
        self.upper = np.array(upper_color, dtype=np.uint8)
        self.name = name

    def __repr__(self):
        return f"ColorRange(name='{self.name}', lower={self.lower.tolist()}, upper={self.upper.tolist()})"
    
    def contains(self, color):
        """Check if a given color (HSV or BGR) lies within the range."""
        color = np.array(color, dtype=np.uint8)
        return np.all(color >= self.lower) and np.all(color <= self.upper)
    
    def apply_mask(self, image, colorspace="HSV"):
        """Return a mask of pixels within this range."""
        if colorspace.upper() == "BGR":
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            hsv = image
        return cv2.inRange(hsv, self.lower, self.upper)
    
    def adjust(self, delta_lower=(0,0,0), delta_upper=(0,0,0)):
        """Adjust the lower/upper bounds by deltas."""
        self.lower = np.clip(self.lower + np.array(delta_lower), 0, 255)
        self.upper = np.clip(self.upper + np.array(delta_upper), 0, 255)

    def expand(self, amount):
        """Expand range by a uniform amount on all channels."""
        self.lower = np.clip(self.lower - amount, 0, 255)
        self.upper = np.clip(self.upper + amount, 0, 255)

    def average_with(self, other):
        """Blend this range with another ColorRange."""
        lower_avg = ((self.lower.astype(np.int32) + other.lower.astype(np.int32)) // 2).astype(np.uint8)
        upper_avg = ((self.upper.astype(np.int32) + other.upper.astype(np.int32)) // 2).astype(np.uint8)
        return ColorRange(lower_avg, upper_avg, name=f"Average({self.name},{other.name})")
    
    def to_dict(self):
        return {
            "name": self.name,
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist()
        }
    
    @staticmethod
    def from_dict(data):
        return ColorRange(data["lower"], data["upper"], data.get("name"))

    def save(self, path):
        """Save range as JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path):
        """Load range from JSON."""
        with open(path, "r") as f:
            return ColorRange.from_dict(json.load(f))