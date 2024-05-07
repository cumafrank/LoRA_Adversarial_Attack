# LoRA_Adversarial_Attack

This project explores the efficacy of Low-Rank Adaptation (LoRA) as a technique to enhance the robustness of ResNet18, a widely used deep neural network architecture. By adapting the fully-connected and convolutional layers of ResNet18 with LoRA, and subjecting the model to adversarial attacks, we aim to assess whether LoRA can mitigate the vulnerabilities of DNNs to adversarial examples.

## Project Structure
```
.
├── Final - Report.pdf
├── Final - Slides.pdf
├── LoRa_Adversarial_Attack.ipynb # Web friendly notebook
├── README.md
├── data.zip
├── data_loader.py
├── eval.py
├── model.py
├── run.py # __main__ code
└── test.py
```

## Project Demo
Go to `https://drive.google.com/drive/folders/1zSIXfaEfNmEL5bNHsBKtgZCBV_aPMUau?usp=drive_link` for video and audio

## Data Repostory
Go to `https://drive.google.com/drive/folders/1zSIXfaEfNmEL5bNHsBKtgZCBV_aPMUau?usp=drive_link` for model checkpoints and ImageNet-tiny dataset we used in the project.

## Methodology
1. Data Preparation
The dataset used for this project is the ImageNet-tiny dataset, which is preprocessed using standard normalization techniques to fit the input requirements of ResNet18.

2. Model Training
The models are trained using stochastic gradient descent with momentum and a learning rate schedule. We explore several configurations of LoRA applied to ResNet18, evaluating their performance on clean and adversarial examples.

3. Evaluation
The models are evaluated based on accuracy, model certainty, and attack success rate against FGSM and PGD adversarial attacks.

## Results
The results of the experiments are summarized in the `Final - Report.pdf` directory. 
![Result/FGSM - Attack success rate.png](https://github.com/cumafrank/LoRA_Adversarial_Attack/blob/main/Result/FGSM%20-%20Attack%20success%20rate.png)
![Result/FGSM - Model Certainty.png](https://github.com/cumafrank/LoRA_Adversarial_Attack/blob/main/Result/FGSM%20-%20Model%20Certainty.png)

The key findings include:

- Model Performance: LoRA enhances the accuracy of ResNet18.
- Model Certainty: The models with LoRA adaptations retain higher certainty in their predictions under adversarial attacks.
- Attack Success Rate: The LoRA-adapted models show improved robustness against adversarial attacks.
