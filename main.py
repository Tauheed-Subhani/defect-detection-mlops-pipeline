from fastapi import FastAPI, File, UploadFile
from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn as nn
import io

# Setup
app = FastAPI()

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()

# Classes
class_names = ['defective', 'ok']

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.get("/")
def root():
    return {"message": "Defect Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return {
        "prediction": class_names[predicted.item()],
        "confidence": round(confidence.item(), 4)
    }