# Fruit Classification with CNNs and Transfer Learning

This repository contains experiments on fruit image classification using two approaches:

1. A custom convolutional neural network (CNN) built from scratch  
2. Transfer learning with pre-trained ResNet models

Both approaches are applied to a “banana vs. other fruits” binary task and a five-class multi-fruit classification task.

---

## Table of Contents

- [Dataset](#dataset)  
- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Data Preparation](#data-preparation)  
- [Training](#training)  
- [Evaluation](#evaluation)  
- [Results](#results)  
- [Dependencies](#dependencies)  
- [License](#license)  

---

## Dataset

We use the [Fruits Classification Dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification), which contains ~10,000 JPEG/PNG images evenly distributed among five classes (strawberry, banana, apple, grape, mango). The original split (97% train / 2% validation / 1% test) is preserved, and for the binary experiments we apply random undersampling to balance the two selected classes.

---

## Project Structure

```
.
├── CNN/
│   ├── binary_CNN.ipynb
│   └── multiclass_CNN.ipynb
├── transfer_learning/
│   ├── resnet_binary.ipynb
│   └── resnet_multiclass.ipynb
├── data_splitting.py
├── requirements.txt
└── README.md
```

- **CNN/**  
  Jupyter notebooks implementing a four-layer CNN from scratch for both binary and multiclass classification.

- **transfer_learning/**  
  Notebooks demonstrating fine-tuning of ResNet-18 and ResNet-50 on our fruit dataset.

- **data_splitting.py**  
  Python script to reorganize raw images into `train/`, `valid/`, and `test/` folders, perform undersampling for binary tasks, and apply resizing/augmentation.

- **requirements.txt**  
  Exact package versions for reproducibility.

---

## Installation

1. Clone this repository  
   ```bash
   git clone https://github.com/yourusername/fruit-classifier.git
   cd fruit-classifier
   ```

2. Create a virtual environment and install dependencies  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Download the Kaggle fruit dataset and place the unzipped folder as `data/fruits/`.

---

## Data Preparation

Run the splitting script to generate `data/train/`, `data/valid/`, and `data/test/`:

```bash
python data_splitting.py --input-dir data/fruits --output-dir data
```

This script will:

- Copy images into the three sets (97% / 2% / 1%)  
- Perform random undersampling for binary tasks  
- Resize images to 128×128 (CNN) or 224×224 (ResNet)  
- Apply normalization and data augmentations during training

---

## Training

Open the relevant notebook based on your task:

- **Binary classification with custom CNN**:  
  `CNN/binary_CNN.ipynb`

- **Multiclass classification with custom CNN**:  
  `CNN/multiclass_CNN.ipynb`

- **Binary classification with ResNet**:  
  `transfer_learning/resnet_binary.ipynb`

- **Multiclass classification with ResNet**:  
  `transfer_learning/resnet_multiclass.ipynb`

Each notebook walks through:

- Model definition  
- Training loop (Adam optimizer, lr=1e-4, batch size=64, early stopping)  
- Saving `best_model.pth`  
- Plotting loss & accuracy curves

---

## Evaluation

After training, evaluation metrics (accuracy, precision, recall, F1-score) and confusion matrices are generated for the test set. Qualitative results include:

- Feature maps from early layers  
- Sample predictions with correct/incorrect labels

Refer to the “Results and Discussion” sections in each notebook for detailed analysis.

---

## Results

- Custom CNN (4 conv layers)  
  - Binary task: 91% accuracy  
  - Multiclass task: ~88% accuracy  

- ResNet-18 Transfer Learning  
  - Binary task: 95% accuracy  
  - Multiclass task: ~92% accuracy  

See the notebooks for full breakdowns and visualizations of errors and feature activations.

---

## Dependencies

- Python 3.11  
- PyTorch 2.2  
- torchvision  
- numpy, pandas, matplotlib  
- scikit-learn  

Exact versions in `requirements.txt`.

---

## License

This project is released under the MIT License.  
Feel free to use and adapt it for your own research or projects!
