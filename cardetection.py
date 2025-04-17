#!/usr/bin/env python
# coding: utf-8

# In[13]:


from PIL import Image
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset



def convert_image_to_numpy(image_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_array = []
    for file in image_files:
        img_path = os.path.join(image_dir, file)
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        image_array.append(img_array)
    return np.transpose(np.stack(image_array), (0, 3, 1, 2))      


training = convert_image_to_numpy(r"data/training_images")
print(training.shape)




training = convert_image_to_numpy(r"data/training_images")
training = torch.tensor(training, dtype=torch.float32) / 255.0 
labels = torch.randint(0, 10, (training.shape[0],)) 

dataset = TensorDataset(training, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)




class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flattened_size = self._get_flattened_size()
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def _get_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)  
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            return x.numel()  

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  
        return x




model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()




torch.save(model.state_dict(), 'model_weights.pth')

