# Human Image Segmentation – Streamlit App

U-Net (ResNet34 encoder) that segments people in images.
Features: overlay, binary mask, transparent cutout, background replace/blur, PNG downloads.

## Run locally
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501

## Deploy on Streamlit Community Cloud (free)
1. Create a GitHub repo and push this folder. The model is ~98 MB, so use Git LFS:
   ```bash
   git init
   git lfs install
   git lfs track "*.pth"        # already in .gitattributes
   git add .
   git commit -m "Human segmentation app"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
2. Go to https://share.streamlit.io → **New app** → pick the repo, branch `main`, main file `app.py` → **Deploy**.

(Alternative: host the .pth on Hugging Face Hub and download it at startup.)

## Files
- `app.py` – the Streamlit app
- `best_human_seg.pth` – trained weights
- `requirements.txt` – CPU PyTorch build to keep the install small
- `.streamlit/config.toml` – upload limit and theme
