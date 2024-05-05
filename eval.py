import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
import numpy as np


norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


def unnormalize(img, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])):
    img = img.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

def normalize(img, mean=np.array(norm_mean), std=np.array(norm_std)):
    # Convert the mean and std to tensors
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(img.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(img.device)
    
    # Use out-of-place operations
    img = (img - mean) / std  # This does not modify `img` inplace
    return img

def fgsm_attack(model, device, data, labels, epsilon, norm_mean, norm_std):
    data.requires_grad = True
    outputs = model(data)
    model.zero_grad()
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data.detach()

def pgd_attack(model, device, data, labels, epsilon, alpha, num_iter, norm_mean, norm_std):
    mean = torch.tensor(norm_mean).view(1, -1, 1, 1).to(device)
    std = torch.tensor(norm_std).view(1, -1, 1, 1).to(device)

    # Clone the data to avoid affecting original data
    original_data = data.clone().detach()

    # Denormalize the data
    data = data * std + mean

    # Initialize perturbed data
    perturbed_data = data.clone().detach()
    perturbed_data.requires_grad = True

    for _ in range(num_iter):
        outputs = model((perturbed_data - mean) / std).to(device)  # Normalize before feeding into the model
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()

        # Apply the perturbation using the sign of the gradients
        with torch.no_grad():  # Make sure to include this to avoid tracking history
            perturbed_data = perturbed_data + alpha * perturbed_data.grad.sign()
            # Ensure the perturbations are clipped to the epsilon neighborhood and within valid image range
            perturbed_data = torch.max(torch.min(perturbed_data, data + epsilon), data - epsilon)
            perturbed_data = torch.clamp(perturbed_data, 0, 1)

        perturbed_data = perturbed_data.detach().requires_grad_(True)  # Detach and reattach requires_grad

    # Normalize the data before returning
    perturbed_data = (perturbed_data - mean) / std
    return perturbed_data.detach()

def get_pretrained_resnet18():
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    return model

def evaluate_attacks(model, device, test_loader, epsilon_fgsm, alpha_pgd, epsilon_pgd, num_iter_pgd,norm_mean,norm_std):
    """Evaluate FGSM and PGD attacks based on correctly predicted examples."""
    model.eval()
    correct_before_attack = 0
    success_count_fgsm = 0
    success_count_pgd = 0
    total_predictions = 0
    model_confidence_fgsm = []
    model_confidence_pgd = []
    model_confidence = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # Evaluate the original model's accuracy on unperturbed data
        with torch.no_grad():
            original_outputs = model(data)
            _, original_preds = original_outputs.max(1)
            correct_before_attack += (original_preds == target).sum().item()

        # Apply attacks only to examples that were originally predicted correctly
        correctly_predicted_mask = (original_preds == target)
        correctly_predicted_data = data[correctly_predicted_mask]
        correctly_predicted_labels = target[correctly_predicted_mask]
        probabilities = F.softmax(original_outputs, dim=1)
        model_confidence.extend(probabilities.max(dim=1)[0].tolist())

        if len(correctly_predicted_data) > 0:
            perturbed_data_fgsm = fgsm_attack(model, device, correctly_predicted_data, correctly_predicted_labels, epsilon_fgsm, norm_mean, norm_std)
            perturbed_data_pgd = pgd_attack(model, device, correctly_predicted_data, correctly_predicted_labels, epsilon_pgd, alpha_pgd, num_iter_pgd, norm_mean, norm_std)

            with torch.no_grad():
                output_fgsm = model(perturbed_data_fgsm)
                output_pgd = model(perturbed_data_pgd)
                probabilities_fgsm = F.softmax(output_fgsm, dim=1)
                probabilities_pgd = F.softmax(output_pgd, dim=1)
                #print(output_fgsm)

                _, preds_fgsm = output_fgsm.max(1)
                _, preds_pgd = output_pgd.max(1)

                success_count_fgsm += (preds_fgsm != correctly_predicted_labels).sum().item()
                success_count_pgd += (preds_pgd != correctly_predicted_labels).sum().item()

                # Model confidence on perturbed data
                model_confidence_fgsm.extend(probabilities_fgsm.max(dim=1)[0].tolist())
                model_confidence_pgd.extend(probabilities_pgd.max(dim=1)[0].tolist())
                #print(model_confidence_fgsm)

    #print(model_confidence_fgsm)
    attack_success_rate_fgsm = success_count_fgsm / correct_before_attack if correct_before_attack > 0 else 0
    attack_success_rate_pgd = success_count_pgd / correct_before_attack if correct_before_attack > 0 else 0
    average_confidence_fgsm = sum(model_confidence_fgsm) / len(model_confidence_fgsm) if model_confidence_fgsm else 0
    average_confidence_pgd = sum(model_confidence_pgd) / len(model_confidence_pgd) if model_confidence_pgd else 0
    average_confidence = sum(model_confidence) / len(model_confidence) if model_confidence else 0

    #print('success_cnt : ',correct_before_attack)
    #print(success_count_pgd)
    #print('test: ',success_count_pgd)

    print(f"Correct Predictions Before Attack: {correct_before_attack}, Average Confidence: {average_confidence:.3f}")
    print(f"FGSM Attack Success Rate: {attack_success_rate_fgsm:.2f}, Average Confidence: {average_confidence_fgsm:.2f}")
    print(f"PGD Attack Success Rate: {attack_success_rate_pgd:.2f}, Average Confidence: {average_confidence_pgd:.3f}")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_pretrained_resnet18()
    model.to(device)
    # Initialize test_loader with your test dataset
    # Define epsilon_fgsm, alpha_pgd, epsilon
