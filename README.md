# ASL Classifier

A deep learning project using PyTorch to classify American Sign Language alphabets from images.

## 📁 Dataset
- [ASL Alphabet Dataset (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

## 🏗️ Model
- CNN or ResNet18-based classifier
- Trained using PyTorch
- Achieved 92% accuracy on Test set and 100% on Validation

## 📦 How to Run

```bash
git clone https://github.com/khaaaleed-5/asl-sign-language-classifier.git
cd asl-sign-language-classifier
pip install -r requirements.txt

# Train
python main.py --train

# Evaluate
python main.py --eval --checkpoint outputs/best_model.pth
