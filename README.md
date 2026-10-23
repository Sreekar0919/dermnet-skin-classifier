# DermNet Skin-Condition Classifier

Transfer learning on the public **DermNet** skin-disease image dataset (2024).

- **Model:** EfficientNet pretrained on ImageNet, with a new head (Dense 1024 → Dense 512 → softmax
  over the DermNet classes).
- **Training:** 10 epochs with the base frozen, then the base is unfrozen and fine-tuned for 50 epochs at
  learning rate 1e-5 (vs. 1e-2 for the head).
- `dermnet_cnn_v1.ipynb`: baseline. `dermnet_cnn_v2.ipynb`: adds a learning-rate scheduler and
  separate accuracy curves before and after fine-tuning.

## Run

Download DermNet (for example from Kaggle) as `data/dermnet.zip`. Each notebook unzips it into
`train/` and `test/` folders.

```bash
pip install tensorflow matplotlib
```

Educational project, not a diagnostic tool.
