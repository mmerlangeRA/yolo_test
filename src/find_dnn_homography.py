import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2

class HomographyModel(nn.Module):
    def __init__(self):
        super(HomographyModel, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Initial dummy forward pass to calculate the size
        self._initialize_fc()

    def _initialize_fc(self):
        dummy_input = torch.zeros(1, 2, 128, 128)
        x = self.pool(torch.relu(self.conv1(dummy_input)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.pool(torch.relu(self.conv5(x)))
        x = self.pool(torch.relu(self.conv6(x)))
        self.flattened_size = x.view(1, -1).size(1)
        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.fc2 = nn.Linear(1024, 8)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.pool(torch.relu(self.conv5(x)))
        x = self.pool(torch.relu(self.conv6(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the model
model = HomographyModel()

# Load pre-trained weights if available
# model.load_state_dict(torch.load('path_to_pretrained_weights.pth'))

# Set the model to evaluation mode
model.eval()

# Function to preprocess the images
def preprocess_images(ref_image, photo_image):
    ref_image = cv2.resize(ref_image, (128, 128))
    h_target, w_target = photo_image.shape
    max_size_img_target = max(h_target, w_target)
    reduction_ratio = 128./max_size_img_target
    canvas = np.zeros_like(ref_image)
    print(f'{w_target},{h_target}')
    h_target*=reduction_ratio
    w_target*=reduction_ratio
    h_target = int(h_target)
    w_target=int(w_target)
    print(f'{w_target},{h_target}')
    if abs(h_target-128)<3:
        h_target=128
    if abs(w_target-128)<3:
        w_target=128
    print(f'{w_target},{h_target}')
    photo_image = cv2.resize(photo_image, (h_target, w_target))
    cv2.imshow("ref_image",ref_image)
    cv2.imshow("photo_image",photo_image)
    canvas[:int(h_target), :int(w_target)] = photo_image
    photo_image = canvas
    stacked_images = np.dstack((ref_image, photo_image))
    stacked_images = np.transpose(stacked_images, (2, 0, 1))  # Change to CxHxW
    return stacked_images.astype(np.float32) / 255.0

# Load your images
""" ref_image = cv2.imread('reference_image.jpg', cv2.IMREAD_GRAYSCALE)
photo_image = cv2.imread('real_life_image.jpg', cv2.IMREAD_GRAYSCALE) """

ref_image = cv2.imread(r'C:\Users\mmerl\projects\yolo_test\src\matching\panneaux\France_road_sign_B21-2.svg.png', cv2.IMREAD_GRAYSCALE)
photo_image = cv2.imread(r'C:\Users\mmerl\projects\yolo_test\blob1.png', cv2.IMREAD_GRAYSCALE)

# Preprocess the images
input_data = preprocess_images(ref_image, photo_image)
input_data = torch.tensor(input_data).unsqueeze(0)  # Add batch dimension

# Move to the same device as the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_data = input_data.to(device)
model = model.to(device)

# Predict the homography
with torch.no_grad():
    predicted_homography = model(input_data).cpu().numpy()[0]

# Reshape to 3x3 homography matrix
H = np.append(predicted_homography, 1).reshape(3, 3)
print(H)

# Apply homography to the real-life photo
height, width = ref_image.shape
warped_image = cv2.warpPerspective(photo_image, H, (width, height))

# Save or display the warped image
cv2.imwrite('warped_image.jpg', warped_image)
cv2.imshow('Warped Image', warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
