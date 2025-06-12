import os
import numpy as np
import cv2 as cv
import math
from itertools import combinations
from tensorflow.keras.models import load_model

# Load pre-trained U-Net model
model = load_model(r"distanceofplot/segmentation_model.h5")

# Note: the satellite image is taken from a height of approx 822 meters in google earth 

img_path = r"distanceofplot/image1.png"
gray = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
original = cv.imread(img_path)

assert gray is not None, "Image not found. Check the path!"

# Resize input image for the model (256x256 or as trained)
input_img = cv.resize(original, (256, 256))
input_norm = input_img / 255.0
input_norm = np.expand_dims(input_norm, axis=0)

# Predict building mask
pred_mask = model.predict(input_norm)[0, :, :, 0]
mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255
mask_resized = cv.resize(mask_bin, (original.shape[1], original.shape[0]))

# Find contours
contours, _ = cv.findContours(mask_resized, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

if len(contours) == 0:
    raise Exception("No contours found!")

# after finding contours
min_area = 1500  # (or experiment with 800, 1000 based on your map zoom level)

building_contours = []
for cnt in contours:
    area = cv.contourArea(cnt)
    if area > min_area:
        building_contours.append(cnt)   

# Draw only filtered contours
cv.drawContours(original, building_contours, -1, (0, 255, 0), 2)

# Known scale from user
scale = 129.93 / 796.58  # meters per pixel ≈ 0.1632
print(f"Using scale: {scale:.4f} meters/pixel")

# Filter small contours (noise)
min_area = 500
filtered_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_area]

print(f"Detected {len(filtered_contours)} potential buildings")

# Draw contours
contour_img = original.copy()
for cnt in filtered_contours:
    cv.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2)

# Final display
cv.imshow("Detected Buildings", contour_img)
cv.waitKey(0)
cv.destroyAllWindows()
