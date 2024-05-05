import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# Import custom modules
from model import ModifiedResNet18Conv, ModifiedResNet18
from data_loader import load_images_and_labels, convert_to_dict, CustomImageDataset
from eval import evaluate_attacks  # Ensure this contains your attack logic
import torch.nn as nn
from torchvision import models

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 5  # Ensure this matches the number trained on

    # Load the pretrained model
    model1_path = 'databest_model_additive_conv1_fc.pth'
    model1 = ModifiedResNet18Conv(num_classes=1000, rank=32, alpha=2)
    model1.load_state_dict(torch.load(model1_path, map_location=device))
    model1.to(device)

    model2_path = 'databest_model_additive_fc.pth'
    model2 = ModifiedResNet18(num_classes=1000, rank=32, alpha=2)
    model2.load_state_dict(torch.load(model2_path, map_location=device))
    model2.to(device)

    model_base_path = 'databest_resnet18.pth'
    model_base = models.resnet18(pretrained=True)  # Use pretrained=False for training from scratch
    model_base.fc = nn.Linear(model_base.fc.in_features, 1000)  # Adjusting the final layer for 1000 classes of ImageNet
    model_base.load_state_dict(torch.load(model_base_path, map_location=device))
    model_base.to(device)

    # Data setup
    dataset_path = 'data'
    images, labels = load_images_and_labels(dataset_path)
    data = convert_to_dict(images, labels)
    dataset = CustomImageDataset(data, transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Attack parameters
    epsilon_fgsm = 0.03  # Example value for FGSM
    alpha_pgd = 0.01     # Step size for PGD
    epsilon_pgd = 0.03   # Perturbation limit for PGD
    num_iter_pgd = 5     # Number of iterations for PGD
    norm_mean = [0.485, 0.456, 0.406]  # Normalization mean
    norm_std = [0.229, 0.224, 0.225]   # Normalization std

    # Evaluate attacks
    results1 = evaluate_attacks(model1, device, dataloader, epsilon_fgsm, alpha_pgd, epsilon_pgd, num_iter_pgd, norm_mean, norm_std)
    print('Resnet + LORA on FC and Conv: \n')
    print(results1)

    results2 = evaluate_attacks(model2, device, dataloader, epsilon_fgsm, alpha_pgd, epsilon_pgd, num_iter_pgd, norm_mean, norm_std)
    print('Resnet + LORA on FC: \n')
    print(results2)

    results_base = evaluate_attacks(model_base, device, dataloader, epsilon_fgsm, alpha_pgd, epsilon_pgd, num_iter_pgd, norm_mean, norm_std)
    print('Resnet(base model): \n')
    print(results_base)

if __name__ == '__main__':
    main()








