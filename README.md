# 🌪️ Hurricane Disaster Damage Assessment using Deep Learning

This project focuses on the detection and classification of hurricane-induced building damage using satellite imagery. Deep learning techniques, including transfer learning and custom CNN architectures, are used to identify and classify damaged vs. undamaged areas from aerial images.

---

## 📁 Project Structure

- `Disaster.ipynb` - Baseline model using **VGG16** with Transfer Learning.
- `Disaster2.ipynb` - VM/remote setup for model training and data processing.
- `Disaster3.ipynb` - Advanced model using **custom CNN + SE (Squeeze-and-Excitation) blocks** for better feature learning and performance.

---

## 📊 Dataset

- **Source**: [Kaggle - Satellite Images of Hurricane Damage](https://www.kaggle.com/datasets/kmader/satellite-images-of-hurricane-damage)
- **Format**: RGB satellite images categorized into:
  - `damage` (buildings affected by hurricane)
  - `no_damage` (intact buildings)

---

## 🧠 Models Used

### ✅ Transfer Learning (Disaster.ipynb)
- **Base Model**: VGG16
- **Top Layers**: Global Average Pooling, Dense, Dropout
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam

### 🚀 Custom CNN Model with SE Block (Disaster3.ipynb)
- **Layers**: Conv2D, MaxPooling, BatchNorm, SE Block, Dense
- **SE Block**: Implements channel-wise attention to improve feature representation.
- **Architecture**: Lightweight and modular, suitable for edge deployment.

---

## 🔄 Data Augmentation
- Performed using `ImageDataGenerator`:
  - Rotation
  - Zoom
  - Flipping
  - Width/Height Shift
  - Shear Transformation

---

## ⚙️ Requirements

```bash
tensorflow
keras
numpy
opencv-python
pillow
matplotlib
pandas
scikit-learn
```

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/hurricane-damage-detection.git
   cd hurricane-damage-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebooks sequentially:
   - `Disaster.ipynb` → VGG16 Transfer Learning
   - `Disaster3.ipynb` → Advanced Model Training

4. Optionally run `Disaster2.ipynb` if you're training on a remote VM.

---

## 📈 Evaluation Metrics
- Accuracy
- Precision / Recall / F1-Score
- Confusion Matrix
- ROC-AUC Curve (optional to add for binary classification)

---

## 📌 Future Improvements
- Deploy model via Flask API or TensorFlow Lite
- Multiclass classification (low, medium, high damage levels)
- Integrate object detection to isolate buildings
- Ensemble learning with other pre-trained networks (ResNet, MobileNet)

---

## 📸 Sample Outputs (Optional)
_Add a few sample prediction images or confusion matrix plots here_

---

## 🤝 Acknowledgments
- IEEE DataPort - Hurricane Harvey Dataset
- Kaggle Contributors
- TensorFlow and Keras Libraries

---

## 📬 Contact
If you have any questions or suggestions, feel free to contact me at:
**your.email@example.com**

---

## 📝 License
This project is licensed under the MIT License - see the `LICENSE` file for details.
