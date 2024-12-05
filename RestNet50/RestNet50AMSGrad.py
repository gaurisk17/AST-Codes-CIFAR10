import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}")

num_classes = 10
batch_size = 128
num_epochs = 50
learning_rate = 0.001  # Learning rate for adaptive methods
betas = (0.9, 0.999)  # Coefficients for computing running averages

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

model = models.resnet50(weights="IMAGENET1K_V2")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=5e-4, amsgrad=True)

train_loss_list = []
accuracy_list = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    train_loss_list.append(train_loss)
    accuracy_list.append(accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}%")

plt.figure(figsize=(10, 6))
plt.plot(train_loss_list, label='AMSGrad')
plt.xlabel("Epochs")
plt.ylabel("Training loss")
plt.title("CIFAR-10 Training Loss for AMSGrad Optimizer")
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(accuracy_list, label='AMSGrad')
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("CIFAR-10 Training Accuracy for AMSGrad Optimizer")
plt.legend(loc='lower right')
plt.show()