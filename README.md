# 🏠 Housing Price Prediction Using Images + Tabular Data

A multimodal deep learning project that predicts house prices by fusing **visual features from house photos** (via a CNN) with **structured tabular data** (via an MLP) into a single regression model.

> **Task 3** — AI/ML Engineering Advanced Internship, DevelopersHub Corporation

---

## 📌 Objective

Most house price models rely only on structured data (bedrooms, area, location, etc.). This project goes further by also learning from **what the house actually looks like**, combining:

- **Tabular data** — numeric/categorical house attributes (Kaggle `train.csv`)
- **Image data** — photos of each house (frontal view, kitchen, bedroom, bathroom)

into one end-to-end trainable model, evaluated using **MAE** and **RMSE**.

---

## 🧠 Approach

| Step | Description |
|---|---|
| 1. Data loading | Load tabular CSV; auto-detect the price (target) and id columns |
| 2. Image matching | Match each house's id to its image file(s) on disk |
| 3. Image preprocessing | Combine up to 4 photos per house into a single 2×2 montage, resized and normalized |
| 4. Tabular preprocessing | Handle missing values, one-hot encode categoricals, scale numeric features |
| 5. Model architecture | CNN branch (image) + MLP branch (tabular) → concatenated ("fused") → regression head |
| 6. Training | Trained with early stopping to avoid overfitting |
| 7. Evaluation | MAE and RMSE on a held-out test split |
| 8. Visualization | Loss curves, predicted vs. actual scatter plot, residuals distribution |

### Model Architecture

```
        Image (64x64x3)                  Tabular Features
              │                                 │
      ┌───────▼────────┐              ┌─────────▼─────────┐
      │  Conv2D + BN     │              │   Dense (16)      │
      │  MaxPooling      │              │   Dense (8)       │
      │  Conv2D + BN     │              └─────────┬─────────┘
      │  MaxPooling      │                        │
      │  Conv2D + BN     │                        │
      │  MaxPooling      │                        │
      │  Flatten          │                        │
      │  Dense (16)       │                        │
      └────────┬─────────┘                        │
               │                                    │
               └───────────────┬────────────────────┘
                                │
                       Concatenate (Fusion)
                                │
                          Dense (8, relu)
                                │
                        Dense (1, linear)  →  Predicted Price
```

---

## 📂 Dataset

| Source | File / Folder | Notes |
|---|---|---|
| Kaggle | `train.csv` | Tabular house attributes + price |
| GitHub | `houses dataset/` | House images |

> Dataset files are not included in this repo (size/licensing) — see [Setup](#-setup--usage) to add your own.

---

## ⚙️ Setup & Usage

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Install dependencies
```bash
pip install tensorflow pandas numpy matplotlib scikit-learn opencv-python pillow
```

### 3. Add your data
Place these in the project root:
- `train.csv`
- `houses dataset/` (folder of house images)

### 4. Run the notebook
Open `Housing_Price_Prediction_Using_Images_COMPLETE.ipynb` and run all cells top to bottom.

The **Configuration cell** at the top auto-detects your target/id columns and image naming convention — edit it only if the printed diagnostics tell you to.

---

## 📊 Results

| Metric | Value |
|---|---|
| MAE  | *see notebook output, Section 11* |
| RMSE | *see notebook output, Section 11* |

Visualizations included in the notebook:
- Training vs. validation loss/MAE curves
- Predicted vs. actual price scatter plot
- Residuals distribution histogram

*(Fill in the actual numbers here after running the notebook on your data, before submitting.)*

---

## 🛠️ Tech Stack

- **Python 3**
- **TensorFlow / Keras** — CNN + multimodal fusion model
- **scikit-learn** — preprocessing, train/test split, evaluation metrics
- **pandas / NumPy** — data handling
- **OpenCV** — image loading & preprocessing
- **Matplotlib** — visualizations

---

## 💡 Key Insights

- Combining multiple photos per house into a single montage lets one compact CNN branch capture visual signal from all available views at once.
- Feature fusion (image + tabular) allows the model to correct tabular-only predictions using visual cues (condition, style, curb appeal) that numbers alone don't capture.
- Model performance is sensitive to how well images are matched to rows — dropped/unmatched rows should be monitored (see notebook diagnostics).

## 🔮 Possible Improvements

- Use a pretrained CNN backbone (MobileNetV2 / ResNet50, transfer learning) instead of training from scratch, given the typically small size of house-image datasets.
- Engineer additional tabular features (e.g. price per square foot, location clustering).
- Train branches separately before fine-tuning the fused model.


