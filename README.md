# 🏢 Building Edge Detection

This project uses a **deep learning U-Net model** to detect the **edges of buildings** from aerial or satellite images.

---

## 📂 Project Structure

- `distanceofplot/` — Folder containing datasets, notebooks, and scripts.
  - `dataset/`
    - `images/` — Input images for segmentation.
    - `masks/` — Ground truth masks for segmentation.
  - `training.ipynb` — Jupyter Notebook used to train the improved model (`segmentation_model.h5`).
  - `unet_model_training.py` — Script for training the initial U-Net model.
  - `border.py` — Main script to perform edge detection.
- `unet_model.h5` — Pre-existing trained U-Net model (less accurate).
- `segmentation_model.h5` — New, improved model trained for better building contour detection.

---

## ⚙️ How to Run

1. **Use the trained model** (`segmentation_model.h5`) for better performance.  
   *(If you prefer, you can still run `training.ipynb` or `unet_model_training.py` to train your own model.)*

2. **Install the required libraries**:

```bash
pip install tensorflow opencv-python numpy
