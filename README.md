# 🧠 Medical Diagnosis using CNN

This project implements a Convolutional Neural Network (CNN) to perform image-based medical diagnosis. It follows a structured modular approach to prepare image data, train the model, and evaluate results using a clean and maintainable codebase.

## 📁 Project Structure

```
medical_diagnosis/
├── cnn_model/
│   └── cnn_model_model.py        # CNN architecture and training logic
├── dataset/
│   ├── train/                    # Training images
│   ├── val/                      # Validation images
│   └── test/                     # Testing images
├── image_data_pipeline/
│   └── code.py                   # Image data preprocessing pipeline
├── README.md                     # Project documentation
```

## ⚙️ Components

### 🧩 `cnn_model/cnn_model_model.py`
- Defines the CNN architecture (e.g., Conv2D, MaxPooling, Flatten, Dense layers)
- Includes model compilation, training, and evaluation routines

### 🔄 `image_data_pipeline/code.py`
- Handles data loading, preprocessing, and augmentation
- Uses tools like `ImageDataGenerator` from Keras or `torchvision` transforms

### 📂 `dataset/train`, `val`, `test`
- Structured image dataset for supervised learning
- Supports model training and validation for medical image classification

## 📊 Features
- Custom CNN architecture for classification
- Clean image pipeline for robust preprocessing
- Modular code for scalability and maintenance
- Easily extendable for new datasets or architectures

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/Vaishnavishatagopam77/medical_diagnosis.git
   cd medical_diagnosis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run data pipeline and train model:
   ```bash
   python image_data_pipeline/code.py
   python cnn_model/cnn_model_model.py
   ```

## 📜 License

This project is licensed under the MIT License.

## 👩‍💻 Author

Vaishnavi Shatagopam
