# Data

This project uses the **Chest X-Ray Pneumonia** dataset from Kaggle
(~5,800 pediatric chest X-rays, binary NORMAL / PNEUMONIA labels).

## Download

1. Create a free Kaggle account and generate an API key
   (kaggle.com → Settings → API → Create New Token → save `kaggle.json`
   to `~/.kaggle/`).
2. Install the Kaggle CLI: `pip install kaggle`
3. Download and unzip:

```bash
mkdir -p data && cd data
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d chest_xray
```

You should end up with `data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/`.

**On Google Colab:** upload your `kaggle.json` when prompted, or mount
Google Drive and point `--data-dir` at the dataset there.

## Going further

Once the binary classifier works, the natural stretch goal is the
**NIH ChestX-ray14** dataset (112,120 images, 14 disease labels) —
available on Kaggle (`nih-chest-xrays/data`) — and extending the model
to multi-label classification, CheXNet-style.
