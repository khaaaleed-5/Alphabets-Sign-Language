import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
import torch
from torch.utils.data import TensorDataset, DataLoader


transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

def get_dataloaders(data_dir, batch_size=64):
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    num_classes = len(dataset.classes)

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, dataset.classes, num_classes


def get_test_dataloaders(test_dir, batch_size=64):
    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing', 'space']
    label_to_index = {label: idx for idx, label in enumerate(class_names)}
    test_images = []
    image_paths = []
    labels = []

    for filename in os.listdir(test_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                label_char = filename[0].upper()
                if label_char in label_to_index:
                    labels.append(label_to_index[label_char])
                    img_path = os.path.join(test_dir, filename)
                    image = Image.open(img_path).convert('RGB')
                    image = transform(image)
                    test_images.append(image)
                    image_paths.append(filename)
                    test_tensor = torch.stack(test_images)

                    label_tensor = torch.tensor(labels)

                    test_dataset = TensorDataset(test_tensor, label_tensor)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, class_names