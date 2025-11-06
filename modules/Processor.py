# Modules
from .types.Settings import Settings
from .types.ColorRange import ColorRange

import numpy as np
import cv2

class ProcessorTemplate:
    def __init__(self, size, ranges : ColorRange, nest_tolerance=0.15):
        self.size = size
        self.ranges = ranges
        self.nest_tolerance = nest_tolerance

    @staticmethod
    def makeHough(circleDp, circleMinDist, circleParam1, circleParam2, circleMinRadius, circleMaxRadius):
        return {
            "method": cv2.HOUGH_GRADIENT,
            "dp": circleDp,
            "minDist": circleMinDist,
            "param1": circleParam1,
            "param2": circleParam2,
            "minRadius": circleMinRadius,
            "maxRadius": circleMaxRadius,
        }

class ProcessCache:
    def __init__(self, kernel = None, houghCache = None, maskBuffer = None, circleMask = None):
        self.kernel = kernel
        self.houghCache = houghCache



class ImageProcessor:
    def __init__(self, template : ProcessorTemplate, settings : Settings):
        self.template = template
        self.settings = settings

        # Cache
        self.cache = ProcessCache(self.generate_kernel(template.size), ProcessorTemplate.makeHough(
            self.settings.circleDp,
            self.settings.circleMinDist,
            self.settings.circleParam1,
            self.settings.circleParam2,
            self.settings.circleMinRadius,
            self.settings.circleMaxRadius
        ))

    # Static

    # Generate Kernel
    @staticmethod
    def generate_kernel(size, base = 3):
        _, w = size
        k = max(3, min(base * (w // 640), 7))
        return np.ones((k, k), np.uint8)

    # Generate Mask
    @staticmethod
    def generate_mask(image, cr : ColorRange, kernel, blur):
        mask = cr.apply_mask(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, blur, 0)
        return mask
    
    # Remove Static Nests
    @staticmethod
    def remove_nested_circles(circles, containment_tol=1.0):
        if not circles:
            return circles
        circles = [(int(x), int(y), int(r)) for x, y, r in circles]
        circles = sorted(circles, key=lambda c: c[2], reverse=True)
        keep = []
        for (x1, y1, r1) in circles:
            contained = False
            for (x2, y2, r2) in keep:
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                if dist_sq <= (r2 - r1 * containment_tol)**2:
                    contained = True
                    break
            if not contained:
                keep.append((x1, y1, r1))
        return keep
    
    # Instance

    def detect_balls(self, image):
        outputs = []
        for cr in self.template.ranges:
            mask = self.generate_mask(image, cr, self.cache.kernel, self.settings.gaussBlur)
            circles = cv2.HoughCircles(mask, **self.cache.houghCache)

            output = []
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for x, y, r in circles[0, :]:
                    circle_mask = np.zeros_like(mask)
                    cv2.circle(circle_mask, (x, y), r, 255, -1)
                    coverage = cv2.countNonZero(cv2.bitwise_and(mask, circle_mask)) / (np.pi * r * r)
                    if coverage > self.settings.coverage_threshold:
                        output.append((x, y, r))

            output = self.remove_nested_circles(output, self.template.nest_tolerance)
            outputs.append(output)

        return outputs