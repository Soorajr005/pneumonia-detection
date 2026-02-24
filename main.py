from fastapi import FastAPI,UploadFile,File
import torch
import torch.nn as nn
from torchvision import models,transforms
import io
from PIL import Image

app = FastAPI()

model = models.resnet18()

a = model.fc.in_features

model.fc = nn.Linear(a,2)

model.load_state_dict(torch.load("pneumonia_resnet18.pth", map_location=torch.device('cpu')))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229, 0.224, 0.225])
])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    image_tensor = preprocess(image).unsqueeze(0)


    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

        classes = ['NORMAL','PNEUMONIA']
        result = classes[predicted.item()]


    return {"Predicted": result}



