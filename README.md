# 🏢 Building Edge Detection

This project uses a **deep learning U-Net model** to detect the **edges of buildings** from aerial or satellite images.  

---

## 📂 Project Structure

- `distanceofplot/` — Folder containing images and code.
- `unet_model_training.py` — Pre-trained U-Net model used for prediction.
- `border.py` — Main script to perform edge detection.

---

## ⚙️ How to Run

1. **Run the pre-trained U-Net model** (`unet_model_training.py`) if you haven't already.  
   The model is required to perform the building edge detection.

2. **Install the required libraries**:

```bash
pip install tensorflow opencv-python numpy