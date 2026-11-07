

# Set thread limits BEFORE importing PyTorch, NumPy or Matplotlib
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# 1. Configure CPU
torch.set_num_threads(2)
device = torch.device("cpu")

print("Using device:", device)


# 2. Load pretrained ResNet-50
print("Loading ResNet-50...")

weights = ResNet50_Weights.DEFAULT

model = resnet50(weights=weights)
model = model.to(device)
model.eval()

preprocess = weights.transforms()

print("Model loaded successfully!")


# 3. Load image
image_path = "images/4.jpg"

image = Image.open(image_path).convert("RGB")
input_tensor = preprocess(image).unsqueeze(0).to(device)

print("Image loaded successfully!")


# 4. Make prediction
print("Making prediction...")

with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)[0]

categories = weights.meta["categories"]

predicted_class = int(probabilities.argmax())
predicted_label = categories[predicted_class]
confidence = probabilities[predicted_class].item()

print("\nPredicted class:", predicted_label)
print(f"Confidence: {confidence:.2%}")


# 5. Display top 5 predictions
top5_prob, top5_indices = torch.topk(probabilities, 5)

print("\nTop 5 predictions:")

for prob, idx in zip(top5_prob, top5_indices):
    print(f"{categories[idx.item()]}: {prob.item():.2%}")


# 6. Generate Grad-CAM
print("\nGenerating Grad-CAM...")

target_layers = [model.layer4[-1]]
targets = [ClassifierOutputTarget(predicted_class)]

with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

print("Grad-CAM generated successfully!")


# 7. Prepare original image for visualization

from torchvision import transforms

display_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])

display_image = display_transform(image)

image_array = np.asarray(display_image, dtype=np.float32) / 255.0
# display_image = weights.transforms()(image)

# # Convert normalized tensor back to displayable RGB
# mean = np.array(weights.meta["mean"])
# std = np.array(weights.meta["std"])

# image_array = display_image.permute(1, 2, 0).numpy()
# image_array = np.clip(image_array * std + mean, 0, 1)


# 8. Display results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_array)
plt.title(f"ResNet-50: {predicted_label}\nConfidence: {confidence:.2%}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image_array)
plt.imshow(grayscale_cam, cmap="jet", alpha=0.45)
plt.title("Grad-CAM Explanation")
plt.axis("off")

plt.tight_layout()


# 9. Save output
os.makedirs("outputs", exist_ok=True)

output_name = os.path.splitext(os.path.basename(image_path))[0]

output_path = f"outputs/{output_name}_gradcam.png"

plt.savefig(output_path, dpi=150, bbox_inches="tight")

print(f"\nResult saved successfully: {output_path}")

plt.show()









































# import os
# import torch
# import matplotlib.pyplot as plt

# from PIL import Image
# from torchvision.models import resnet50, ResNet50_Weights
# from torchvision import transforms
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# # 1. Load pretrained ResNet-50
# weights = ResNet50_Weights.DEFAULT
# # model = resnet50(weights=weights)
# # model.eval()
# device = torch.device("cpu")

# model = resnet50(weights=weights)
# model = model.to(device)
# model.eval()
# # Use the preprocessing expected by this model
# preprocess = weights.transforms()

# # 2. Load your image
# image_path = "images/b3.jpg"
# image = Image.open(image_path).convert("RGB")
# input_tensor = preprocess(image).unsqueeze(0)

# # 3. Make a prediction
# # with torch.no_grad():
# #     output = model(input_tensor)
# #     probabilities = torch.softmax(output, dim=1)[0]

# # predicted_class = int(probabilities.argmax())
# # # Show top 5 predictions
# # top5_prob, top5_indices = torch.topk(probabilities, 5)

# # print("\nTop 5 predictions:")

# # for prob, idx in zip(top5_prob, top5_indices):
# #     print(f"{categories[idx]}: {prob.item():.2%}")

# # categories = weights.meta["categories"]
# # predicted_label = categories[predicted_class]
# # confidence = probabilities[predicted_class].item()

# # print("Predicted class:", predicted_label)
# # print(f"Confidence: {confidence:.2%}")
# # 3. Make a prediction
# with torch.no_grad():
#     output = model(input_tensor)
#     probabilities = torch.softmax(output, dim=1)[0]

# categories = weights.meta["categories"]

# predicted_class = int(probabilities.argmax())
# predicted_label = categories[predicted_class]
# confidence = probabilities[predicted_class].item()

# print("Predicted class:", predicted_label)
# print(f"Confidence: {confidence:.2%}")

# # Show top 5 predictions
# top5_prob, top5_indices = torch.topk(probabilities, 5)

# print("\nTop 5 predictions:")

# for prob, idx in zip(top5_prob, top5_indices):
#     print(f"{categories[idx.item()]}: {prob.item():.2%}")

# # 4. Generate Grad-CAM for the predicted class
# target_layers = [model.layer4[-1]]
# targets = [ClassifierOutputTarget(predicted_class)]

# with GradCAM(model=model, target_layers=target_layers) as cam:
#     grayscale_cam = cam(
#         input_tensor=input_tensor,
#         targets=targets
#     )[0]

# # 5. Prepare the image for displaying the heatmap
# display_transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224)
# ])

# display_image = display_transform(image)
# image_array = (torch.tensor(list(display_image.getdata()))
#                .reshape(224, 224, 3).numpy() / 255.0)

# # 6. Display original image and Grad-CAM heatmap
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(image_array)
# plt.title(f"ResNet-50: {predicted_label}")
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.imshow(image_array)
# plt.imshow(grayscale_cam, cmap="jet", alpha=0.45)
# plt.title("Grad-CAM Heatmap")
# plt.axis("off")

# plt.tight_layout()

# # 7. Save the result
# os.makedirs("outputs", exist_ok=True)
# plt.savefig("outputs/cat_gradcam.png", dpi=150)
# plt.show()

# print("Grad-CAM image saved in outputs/cat_gradcam.png")