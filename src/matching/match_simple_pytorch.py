import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from scipy.spatial.distance import cdist
import pickle

# Load the VGG16 model pretrained on ImageNet without the top classification layer
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(self.vgg16.features.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        return x

model = VGG16FeatureExtractor()
model.eval()  # Set the model to evaluation mode

# Preprocessing function
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(img_path, model):
    print(f"extracting {img_path}")
    img = Image.open(img_path).convert('RGB')
    img_data = preprocess(img)
    img_data = img_data.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(img_data)
    print(features.shape)
    return features.flatten().numpy()

reference_features = []
reference_image_paths = []

def setup_references(folder_path:str,save_name = "reference_data.pk"):
    global reference_features, reference_image_paths
    print("setup_references")
    save_path = os.path.join(folder_path,save_name)
    if os.path.exists(save_path):
        print(f'Loading pickle from {save_path}')
        with open(save_path, 'rb') as f:
            reference_features, reference_image_paths = pickle.load(f)
    else:
        reference_folder = folder_path
        for filename in os.listdir(reference_folder):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                reference_image_path = os.path.join(reference_folder, filename)
                features = extract_features(reference_image_path, model)
                reference_features.append(features)
                reference_image_paths.append(reference_image_path)
        
        with open(save_path, 'wb') as f:
            print(f'saving pickle at {save_path}')
            pickle.dump((reference_features, reference_image_paths), f)
    return reference_features, reference_image_paths

def find_matching(query_image_path, reference_features,reference_image_paths, verbose=False):
    global model
    print(f'query_image_path is {query_image_path}')
    # Extract features for the query image
    #print(model)
    query_features = extract_features(query_image_path, model)
    # Compute distances between query image features and reference image features
    distances = cdist([query_features], reference_features, 'cosine')[0]

    if verbose:
        index = 0
        for distance in distances:
            print(f'{reference_image_paths[index]}  {distance}')
            index += 1
    # Find the best match
    best_match_idx = np.argmin(distances)

    if verbose:
        best_match_distance = distances[best_match_idx]
        print(f'Best match: {reference_image_paths[index]} with distance: {best_match_distance}')

    return best_match_idx


#find_matching("query3.jpg",reference_features,reference_image_paths)
