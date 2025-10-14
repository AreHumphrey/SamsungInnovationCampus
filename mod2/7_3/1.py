import torch
import torchvision.transforms as T
from torchvision.models import resnet18
from PIL import Image
import os
import pandas as pd

# Загрузка модели
model = resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('best_model.pth'))  # после обучения
model.eval()

# Трансформации
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dir = 'test'
ids = []
labels = []

for img_name in sorted(os.listdir(test_dir)):
    img_id = img_name.split('.')[0]
    img_path = os.path.join(test_dir, img_name)
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        pred = model(x).softmax(1).argmax(1).item()

    label = 'cleaned' if pred == 0 else 'dirty'  # уточните порядок меток!
    ids.append(img_id)
    labels.append(label)

pd.DataFrame({'id': ids, 'label': labels}).to_csv('predict.csv', index=False)