import torchvision.transforms as transforms

def get_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # or 3 if your model expects RGB
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])