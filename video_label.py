#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 19:19:15 2024

@author: tylerbarkin
"""

import cv2
import pandas as pd
import torch

torch.set_default_device('mps')
# Path to your video file
video_path = 'Ex1_short.mp4'
#%%

def label_video(video_path):
    cap = cv2.VideoCapture(video_path)
    current_label = 0.0  # No label initially
    labels = []  # Store (frame_number, label)
    
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Display the frame
        cv2.imshow('Frame', frame)
        
        # Play video continuously, wait 25ms for a key press
        key = cv2.waitKey(25) & 0xFF
        if key == ord('n'):
            # Temporarily stop video for new label entry
            # cv2.waitKey(0)  # Wait indefinitely until any key is pressed
            new_label = input("Enter the new label (percentage): ")
            current_label = float(new_label)
        elif key == ord('q'):
            break  # Quit
        
        # Save the current frame number and label
        if current_label is not None:
            labels.append((frame_number, current_label))
        
        frame_number += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    return labels


labels = label_video(video_path)

# Optionally, save labels to a CSV file
labels_df = pd.DataFrame(labels, columns=['FrameNumber', 'Label'])
labels_df.to_csv('video_labels.csv', index=False)

print("Labeling completed and saved to video_labels.csv")

