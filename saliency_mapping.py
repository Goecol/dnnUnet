pip install torch torchvision torchcam matplotlib
import torch
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image # Import to_pil_image

# Using the trained UNet Model
model = UNet(num_of_classes = 1)  # UNet Model Class
model.load_state_dict(torch.load("my_trained_model.pth")) #Loading the trained model
model.eval()  # Set the model to evaluation mode

#Preprocessing the input image

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust to the input size your model requires
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example for ImageNet
])

# Loading the image and applying the transformations
image = Image.open('path_to_image.jpg')
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension


# Selecting the last convolutional layer for Grad-CAM
target_layer = model.outconv  # Last convolutional layer of AlexNet

# Initializing Grad-CAM
cam_extractor = GradCAM(model, target_layer)

# Forward pass through the model
output = model(input_tensor)

# Instead of argmax on the entire output, get argmax for each pixel in the output
# assuming the class dimension is 1
predicted_class = output.argmax(dim=1)


#Selecting specific pixel or average for GradCAM
average_predicted_class = predicted_class.type(torch.float32).mean().int().item()

# Extracting Grad-CAM heatmap for the predicted class
heatmap = cam_extractor(average_predicted_class, output)

# Convert the heatmap tensor to a PIL Image
heatmap_pil = to_pil_image(heatmap[0])

# Overlay the heatmap on the image
result = overlay_mask(image, heatmap_pil, alpha=0.5) # Pass heatmap_pil instead of heatmap[0]

# Display the original image, heatmap, and overlay
plt.figure(figsize=(12, 4))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

# Heatmap
plt.subplot(1, 3, 2)
# Squeeze the heatmap to remove the extra dimension before displaying
plt.imshow(heatmap[0].squeeze(), cmap='jet')
plt.title("Grad-CAM Heatmap")
plt.axis('off')

# Overlay
plt.subplot(1, 3, 3)
plt.imshow(result)
plt.title("Overlay")
plt.axis('off')

plt.tight_layout()
plt.show()