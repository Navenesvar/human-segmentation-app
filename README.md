# 🧍 Human Image Segmentation

A Streamlit web app that detects and segments people in images using a U-Net model with a ResNet34 encoder. Upload a photo and get back a mask, a colored overlay, a transparent cutout, or a version with the background replaced.

🔗 **Live demo:** [https://navenesvar-human-segmentation-app-app-wa1aux.streamlit.app/](https://navenesvar-human-segmentation-app-app-wa1aux.streamlit.app/)

**Validation Dice score: ~0.96**

> ⏳ The app is hosted on Streamlit Community Cloud's free tier. If it has been idle, it may be asleep. Click **"Yes, get this app back up!"** and wait a few seconds.

---

## ✨ Features

- Upload JPG, PNG or WEBP images
- Four output modes:
  - **Overlay**: colored highlight over the person (adjustable color and opacity)
  - **Binary mask**: black-and-white segmentation mask
  - **Cutout**: person on a transparent background
  - **Replace background**: solid color or blur
- Adjustable mask threshold (default 0.50, the best value found on validation data)
- Download the result and the mask as PNG
- Runs on CPU, and uses a GPU automatically if one is available

---

## 🧠 Model

| Item | Details |
|---|---|
| Architecture | U-Net (`segmentation_models_pytorch`) |
| Encoder | ResNet34 (ImageNet pretrained at training time) |
| Input size | 320 × 320 |
| Output | 1-channel probability map (person vs. background) |
| Loss | 0.4 · BCE + 0.4 · Dice + 0.2 · Focal |
| Optimizer | AdamW with ReduceLROnPlateau |
| Dataset | [Human Segmentation Dataset](https://github.com/VikramShenoy97/Human-Segmentation-Dataset) (80/20 train/val split) |
| Best val Dice | 0.9595 |

Training code is in the original notebook (`human_image_segmentation.ipynb`).

---

## 📁 Project Structure

```
human-segmentation-app/
├── app.py                  # Streamlit application
├── best_human_seg.pth      # Trained model weights (Git LFS)
├── requirements.txt        # Python dependencies (CPU PyTorch)
├── .streamlit/
│   └── config.toml         # Upload limit and theme
├── .gitattributes          # Git LFS tracking for *.pth
└── README.md
```

---

## 🚀 Run Locally

**Requirements:** Python 3.9 – 3.12

```bash
# 1. Clone the repo
git clone https://github.com/navenesvar/human-segmentation-app.git
cd human-segmentation-app

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
streamlit run app.py
```

The app opens at http://localhost:8501.

> **Note:** The model file is stored with Git LFS. Install it from https://git-lfs.com and run `git lfs install` before cloning, otherwise `best_human_seg.pth` will be a tiny pointer file and the model won't load.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository (with Git LFS enabled for `*.pth`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** and choose your repository.
4. Set the branch to `main` and the main file path to `app.py`.
5. Click **Deploy**. The first build takes a few minutes while PyTorch installs.

---

## 🛠️ How It Works

1. The uploaded image is resized to 320 × 320 and normalized with ImageNet mean and std.
2. The U-Net outputs a probability map, which is passed through a sigmoid.
3. The probability map is upsampled to the original image size.
4. It is thresholded (default 0.50) to create the mask.
5. The mask is used to build the overlay, cutout or background replacement.

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `Could not load the model` or an invalid checkpoint error | The `.pth` file is probably a Git LFS pointer. Install Git LFS and run `git lfs pull`. |
| App shows a "wake up" screen | Free-tier apps sleep when idle. Click the button to restart it. |
| Slow first run | The model loads once and is then cached. Later predictions are faster. |
| Out-of-memory on a hosted free tier | Upload a smaller image. Images are capped at 2000 px on the longest side. |
| Rough mask edges | Adjust the threshold slider in the sidebar. |

---

## 📦 Tech Stack

[Streamlit](https://streamlit.io) · [PyTorch](https://pytorch.org) · [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) · [OpenCV](https://opencv.org) · [Pillow](https://python-pillow.org)

---

## 🙏 Acknowledgements

- Dataset: [VikramShenoy97/Human-Segmentation-Dataset](https://github.com/VikramShenoy97/Human-Segmentation-Dataset)
- Model library: [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
