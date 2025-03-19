import torch
from src.dataset import get_test_dataloaders
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch.nn as nn
from src.model import get_model




def evaluate_model(checkpoint_path, test_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_loader, classes = get_test_dataloaders(test_dir)
    best_weights = "/outputs/best_model.pth"
    model = get_model(len(classes)).to(device)
    # Load best weights
    model.load_state_dict(torch.load(best_weights))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(xticks_rotation=90, cmap='viridis')
    plt.title("Confusion Matrix")
    plt.show()