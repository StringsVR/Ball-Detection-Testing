import cv2
from modules.Processor import ProcessorTemplate, ImageProcessor
from modules.RangeFinder import RangeFinder
from modules.types.Settings import Settings

if __name__ == "__main__":
    image = cv2.imread("images/IMG_3052.webp")
    image = cv2.resize(image, ((int)(640 * 0.8), (int)(480 * 0.8)))
    rf = RangeFinder()
    color_ranges = rf.find_color_ranges(image)

    template = ProcessorTemplate(
        size=image.shape[:2],
        ranges=[color_ranges["green"], color_ranges["purple"]],
        nest_tolerance=0.05
    )

    settings = Settings(
        (5, 5),
        0.8,
        1.2,
        20,
        50,
        16,
        6,
        100
    )

    processor = ImageProcessor(template, settings)
    colors = processor.detect_balls(image)

    # Draw detected circles
    output_img = image.copy()
    for color in colors:
        for x, y, r in color:
            cv2.circle(output_img, (x, y), r, (0, 255, 0), 2)

    # Show result
    cv2.imshow("Detected Circles", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()