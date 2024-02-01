#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 23:10:59 2024

@author: tylerbarkin
"""

import cv2
import pandas as pd
import math

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Check if MPS is available
torch.set_default_device('mps')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# Path to your video file
video_path = 'Ex1_short.mp4'
csv_file_path = 'video_labels.csv'
labels_df = pd.read_csv(csv_file_path)
batch_size = 32
#%%

class TissueEstimationModel(nn.Module):
    def __init__(self):
        super(TissueEstimationModel, self).__init__()
        # Use a pre-trained model
        base_model = models.resnet18()
        
        # Remove the final fully connected layer
        num_features = base_model.fc.in_features
        base_model = nn.Sequential(*list(base_model.children())[:-1])  # Remove last layer
        
        self.base_model = base_model
        self.fc = nn.Linear(num_features, 1)  # Output one value

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        return self.fc(x)

model = TissueEstimationModel()
model.to(device)  # Move the model to MPS device
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#%%


class VideoFrameDataset(Dataset):
    def __init__(self, video_path, labels_df, transform=None):
        self.video_path = video_path
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        frame_number = self.labels_df.iloc[idx, 0]  # first column is frame number
        label = self.labels_df.iloc[idx, 1].astype(float)  # Convert label to float

        cap = cv2.VideoCapture(self.video_path)
        cap.set(1, frame_number)  # 1 denotes CV_CAP_PROP_POS_FRAMES
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Failed to read frame from video.")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        if self.transform:
            frame = self.transform(frame)

        return frame, torch.tensor(label, dtype=torch.float32)

# Assuming 'video_path' and 'labels_df' are already defined
# Transformations (example, adjust as needed)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Adjust size to match model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet norms
])

# Dataset and DataLoader
dataset = VideoFrameDataset(video_path, labels_df, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#%%
# # Testing the dataloader
# import matplotlib.pyplot as plt

# # Function to display a batch of images
# def show_batch(images, labels):
#     plt.figure(figsize=(12, 8))
#     for i, (image, label) in enumerate(zip(images, labels)):
#         plt.subplot(4, 8, i + 1)
#         plt.imshow(image.permute(1, 2, 0))  # Reorder dimensions for matplotlib
#         plt.title(f"Label: {label.item()}")
#         plt.axis('off')
#     plt.show()

# # Testing the DataLoader
# for batch in dataloader:
#     frames, labels = batch
#     frames = frames.to(device)  # Move data to MPS device
#     labels = labels.to(device).float()
#     print("Batch of frames shape:", frames.shape)
#     print("Batch of labels shape:", labels.shape)

#     # Show a batch of images
#     show_batch(frames, labels)

#     # Break after the first batch for testing
#     break

#%%
# Training loop
num_epochs = 1

# Calculate total number of batches
total_batches = math.ceil(len(dataset) / batch_size)

for epoch in range(num_epochs):
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        
        frames, labels = batch
        frames = frames.to(device)  # Move data to MPS device
        # Convert labels to FloatTensor
        labels = labels.to(device).float()


        # Forward pass
        optimizer.zero_grad()
        outputs = model(frames)

        # Reshape the outputs
        outputs = outputs.squeeze()

        # Loss calculation and backward pass
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_count}/{total_batches}], Loss: {loss.item()}")


# Save model
torch.save(model.state_dict(), 'model.pth')

print("Training complete.")


#%% Validation
# Load validation labels
val_labels_df = pd.read_csv('video_labels.csv')

# Create validation dataset and DataLoader
val_dataset = VideoFrameDataset('Ex1_short.mp4', val_labels_df, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

#%%
# Fetch one batch of validation data
val_frames, val_labels = next(iter(val_dataloader))

# Ensure the model is in evaluation mode
model.eval()

# Move data to the same device as the model
val_frames = val_frames.to(device)
with torch.no_grad():  # No need to track gradients
    val_outputs = model(val_frames).squeeze()

import matplotlib.pyplot as plt

def show_predictions(images, true_labels, predictions):
    plt.figure(figsize=(15, 10))
    for i, (image, true_label, pred_label) in enumerate(zip(images, true_labels, predictions)):
        plt.subplot(4, 8, i + 1)  # Adjust grid size based on your batch_size
        plt.imshow(image.permute(1, 2, 0).cpu().numpy())  # Convert tensor image to numpy
        plt.title(f"True: {true_label.item()}\nPred: {pred_label.item():.2f}")
        plt.axis('off')
    plt.show()

# Convert predictions to the CPU for plotting
val_predictions = val_outputs.to('cpu')

# Display the images with labels and predictions
show_predictions(val_frames, val_labels, val_predictions)


#%%
# Function to perform validation
def validate(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # Disable gradient computation
        for frames, labels in dataloader:
            frames = frames.to(device)
            labels = labels.to(device).float()

            outputs = model(frames)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Perform validation
val_loss = validate(model, val_dataloader)
print(f"Validation Loss: {val_loss}")


