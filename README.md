# Medical Image Denoising Using Convolutional Autoencoders

This project implements the denoising autoencoder described in the paper **“Medical Image Denoising Using Convolutional Denoising Autoencoders”** and compares its performance against median filtering on a dental X‑ray dataset.


## 📁 Project Structure

```
medical-image-denoising/
├── notebooks/
│   └── denoising_autoencoder.ipynb   # Kaggle/Colab notebook implementing the method
├── data/
│   └── Dataset/                      # Grayscale dental X‑ray images (.jpg)
├── results/
│   ├── figures/                      # Sample denoising visualizations
│   └── metrics.txt                   # PSNR comparison results
└── README.md                         # This file
```



## ⚙️ Requirements

* Python 3.7+
* TensorFlow / Keras
* numpy, pandas
* matplotlib, OpenCV

Install via pip:

```bash
pip install tensorflow numpy pandas matplotlib opencv-python
```



## 📂 Data

The dataset consists of 120 grayscale dental X‑ray images located in `data/Dataset/`. Example files:

```
/kaggle/input/medical-image-dataset/Dataset/1.jpg
/kaggle/input/medical-image-dataset/Dataset/2.jpg
…
```

Images are resized to 64 × 64 pixels and normalized to \[0,1].


## 🧠 Model Architecture

A convolutional denoising autoencoder with symmetric encoder–decoder layers:

* **Encoder**: two Conv2D+ReLU blocks with 64 filters, followed by MaxPooling
* **Decoder**: two Conv2D+ReLU blocks with UpSampling, ending in a single-channel sigmoid output
* **Loss**: binary crossentropy

Model summary:

```
Input: (64,64,1)
Conv1 → pool1 → Conv2 → pool2 → Conv3 → upsample1 → Conv4 → upsample2 → Conv5
Output: (64,64,1)
```



## 🚀 Training

1. **Noise injection**: Gaussian noise (σ=1) scaled by 0.07 is added to clean images.
2. **Train/test split**: first 100 images for training, last 20 for validation.
3. **Hyperparameters**: 40 epochs, batch size 10, Adam optimizer.
4. **Early stopping**: patience=10 on validation loss.

Notebook cell example:

```python
model.fit(x_noisy, x_noisy,
          epochs=40,
          batch_size=10,
          validation_data=(x_test_noisy, x_test_noisy),
          callbacks=[EarlyStopping(...)] )
```



## 📊 Evaluation

* **Visual comparison**: plots of original, noisy, denoised (autoencoder), and median‑filtered images.
* **Quantitative metric**: Peak Signal‑to‑Noise Ratio (PSNR). Example:

  * Autoencoder: 69.89 dB
  * Median filter: 58.46 dB

Compute PSNR:

```python
def PSNR(orig, denoised):
    mse = np.mean((orig - denoised)**2)
    return 20 * log10(255.0 / sqrt(mse))
```



## ▶️ Usage

1. Open `denoising_autoencoder.ipynb` in Kaggle or Colab.
2. Ensure the dataset is available under `data/Dataset/`.
3. Run all cells to preprocess, train, and evaluate.
4. View sample outputs in the `results/figures/` folder.



## 📖 References

* The original paper: **Medical Image Denoising Using Convolutional Denoising Autoencoders**.
* Median filtering baseline: OpenCV `cv2.medianBlur` with kernel size 5.
