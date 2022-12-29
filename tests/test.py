# -*- coding: utf-8 -*-
import time
import cv2
import numpy as np
import pytlsd
from skimage.transform import pyramid_reduce

NOTDEF = -1024.0


def get_thresholded_grad(resized_img):
    modgrad = np.full(resized_img.shape, NOTDEF, np.float64)
    anglegrad = np.full(resized_img.shape, NOTDEF, np.float64)

    # A B
    # C D
    A, B, C, D = resized_img[:-1, :-1], resized_img[:-1, 1:], resized_img[1:, :-1], resized_img[1:, 1:]
    gx = B + D - (A + C)  # horizontal difference
    gy = C + D - (A + B)  # vertical difference

    threshold = 5.2262518595055063
    modgrad[:-1, :-1] = 0.5 * np.sqrt(gx ** 2 + gy ** 2)
    anglegrad[:-1, :-1] = np.arctan2(gx, -gy)
    anglegrad[modgrad <= threshold] = NOTDEF
    return gx, gy, modgrad, anglegrad


def draw_distance_field(grandnorm):
    norm_img = 255 * grandnorm / np.max(grandnorm)
    color_map = cv2.applyColorMap(norm_img.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    cv2.namedWindow(f"Distance field", cv2.WINDOW_NORMAL)
    cv2.imshow(f"Distance field", color_map)


def draw_angle_field(anglegrad):
    angle_img = 255 * (anglegrad - np.min(anglegrad)) / (np.max(anglegrad) - np.min(anglegrad))
    color_map = cv2.applyColorMap(angle_img.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    cv2.namedWindow(f"Angle field", cv2.WINDOW_NORMAL)
    cv2.imshow(f"Angle field", color_map)


def draw_segments(gray, segments):
    img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for segment in segments:
        cv2.line(img_color, (int(segment[0]), int(segment[1])), (int(segment[2]), int(segment[3])), (0, 255, 0))

    cv2.namedWindow(f"Detected segments N {len(segments)}", cv2.WINDOW_NORMAL)
    cv2.imshow(f"Detected segments N {len(segments)}", img_color)


gray = cv2.imread('../resources/ai_001_001.frame.0000.color.jpg', cv2.IMREAD_GRAYSCALE)
flt_img = gray.astype(np.float64)

scale_down = 0.8
resized_img = pyramid_reduce(flt_img, 1 / scale_down, 0.6)

# Get image gradients
gx, gy, gradnorm, gradangle = get_thresholded_grad(resized_img)

start = time.perf_counter_ns()
segments = pytlsd.lsd(resized_img, 1.0, gradnorm=gradnorm, gradangle=gradangle)
end = time.perf_counter_ns()
print(f"Lsd time: {(end - start) / 1e6} ms")
segments /= scale_down

gradangle[gradangle == NOTDEF] = -5
draw_distance_field(gradnorm)
draw_angle_field(gradangle)
draw_segments(gray, segments)
cv2.waitKey(0)
