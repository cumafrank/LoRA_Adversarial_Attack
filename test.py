import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
from data_loader import CustomImageDataset, transform, load_images_and_labels, split_data, convert_to_dict
import matplotlib.pyplot as plt

def load_model(model_path, num_classes, device):
    """
    Loads a trained model from a specified path with adaptation for num_classes.
    """
    # Load a pre-trained ResNet model with the original number of classes
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 1000)  # Original number of classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Adapt the fully connected layer to new number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    return model

def evaluate_model(model, test_loader, device):
    """
    Evaluates the model on the test set and visualizes the first 10 predictions.
    """
    model.eval()
    correct = 0
    total = 0
    fig = plt.figure(figsize=(25, 5))  # Define figure size
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader, start=1):
            if i > 10:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            ax = fig.add_subplot(2, 5, i)
            ax.imshow(images[0].cpu().numpy().transpose((1, 2, 0)))
            ax.set_title(f"True: {labels[0].item()}, Pred: {predicted[0].item()}")
            ax.axis('off')

    plt.show()
    return 100 * correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_base_path = 'databest_resnet18.pth'
    
    # Define the path to your dataset directory
    dataset_path = 'data'  # Adjust this to where your dataset is located

    # Load images and labels, prepare test dataset
    images, labels = load_images_and_labels(dataset_path)
    _, _, X_test, _, _, y_test = split_data(images, labels)  # Adjust function if different
    test = convert_to_dict(X_test, y_test)  # Convert test data to dictionary format
    test_dataset = CustomImageDataset(test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the trained model

    model_base = load_model(model_base_path, 5, device)

    # Evaluate the model
    test_accuracy = evaluate_model(model_base, test_loader, device)
    print(f'Test Accuracy: {test_accuracy}%')

if __name__ == '__main__':
    main()