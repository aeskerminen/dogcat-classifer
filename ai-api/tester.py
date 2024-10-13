import argparse
import torch
from torchvision import transforms
from PIL import Image
from model import SimpleCNN  # Ensure the model is available

def load_model(model_path, device):
    """Load a trained model."""
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def process_image(image_path):
    """Preprocess the image from the given path."""
    image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
    image = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict_image(image_path, model, device):
    """Predict if the image is a cat or a dog."""
    image = process_image(image_path).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_names = ['cat', 'dog']
        return class_names[predicted.item()]

def predict(model_path, image_path):
    # Determine if CUDA is available and use it if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the model and predict the image
    model = load_model(model_path, device)
    prediction = predict_image(image_path, model, device)
    print(f'The image is predicted to be a: {prediction}')
    return prediction

def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Load a CNN model and predict the class of an image.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image for prediction')
    args = parser.parse_args()

    predict(args.model_path, args.image_path)


if __name__ == '__main__':
    main()
