from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch import nn as nn
import torchvision.models as models


# dataset=ImageFolder(
#     root="archive\\casting_data\\casting_data\\train",
#     transform=transforms.ToTensor()
# )
# dataLoader=DataLoader(dataset,batch_size=32,shuffle=True)
# image,label=next(iter(dataLoader))
# print(image.shape)
# print(label.shape)

model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 2)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# for epoch in range(5):
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     for images, labels in dataLoader:
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         predicted = outputs.argmax(dim=1)
#         correct += (predicted == labels).sum().item()
#         total += labels.size(0)

#     accuracy = 100 * correct / total
#     print(f"Epoch {epoch+1}: Loss={running_loss/len(dataLoader):.4f}, Accuracy={accuracy:.2f}%")
test_dataset = ImageFolder(
    root="archive/casting_data/casting_data/test",
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")