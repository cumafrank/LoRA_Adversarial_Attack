import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


def load_images_and_labels(root_dir, target_size=(224, 224), max_images_per_class=370):
    images, labels = [], []
    label_dict = {}
    current_label = 0

    for dirname in sorted(os.listdir(root_dir)):
        dirpath = os.path.join(root_dir, dirname)
        # Check if the path is actually a directory
        if os.path.isdir(dirpath):
            if dirname not in label_dict:
                label_dict[dirname] = current_label
                current_label += 1

            class_images = []
            for filename in sorted(os.listdir(dirpath)):
                if filename.endswith('.JPEG'):
                    filepath = os.path.join(dirpath, filename)
                    try:
                        image = Image.open(filepath)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')  # Convert to RGB
                        image = image.resize(target_size)
                        image = np.array(image, dtype=np.uint8)
                        class_images.append(image)
                        if len(class_images) == max_images_per_class:
                            break
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")

            images.extend(class_images)
            labels.extend([label_dict[dirname]] * len(class_images))

    return np.array(images), np.array(labels)


def split_data(images, labels, val_size=350, test_size=250, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=val_size + test_size, stratify=labels, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size, stratify=y_temp, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def convert_to_dict(images, labels):
    return [{'img': img, 'label': label} for img, label in zip(images, labels)]


class CustomImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_data = self.data[idx]
        image = Image.fromarray(image_data['img'])
        label = image_data['label']

        if self.transform:
            image = self.transform(image)

        return image, label

# Example transforms and dataset usage
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])