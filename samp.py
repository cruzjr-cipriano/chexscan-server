from flask import Flask, request, jsonify
from torchvision import transforms
from flask_cors import CORS
from PIL import Image
import torch
from torchvision.models import resnet18
import io
import base64

app = Flask(__name__)
# Add CORS headers for all routes
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST'
    return response

# Load the model architecture
model = resnet18(pretrained=True)
num_classes = 3  # Assuming you have 3 classes (Normal, Pneumonia, Tuberculosis)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('model.pth'))
model.eval()

@app.route('/test')
def get_home():
    return "Hello World"

def transform_image(image):
    # Resize the image to 224x224 pixels
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # Apply transformations
    image_tensor = transform(image)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    # Add channel dimension (RGB)
    image_tensor = image_tensor.expand(-1, 3, -1, -1)
    # Normalize the image
    image_tensor = transforms.functional.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image_tensor

class_names = ['Normal', 'Pneumonia', 'Tuberculosis']

# Function to predict the class of an input image
def predict_image_base64(image_data, model):
    image_bytes = base64.b64decode(image_data)  # Decode the base64 encoded image data into bytes
    image = Image.open(io.BytesIO(image_bytes)) # Convert the bytes into a PIL Image
    image_tensor = transform_image(image)       # Transform the image
    
    with torch.no_grad():       # Make prediction using the model
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]  # Use class_names to get class label
    return predicted_class

@app.route('/predict', methods=['POST'])
def upload_image():
    image_data = request.json.get('image')    # Get the base64 encoded image data from the request
    image_filter = image_data[image_data.index(',')+1:]     # Predict the class of the uploaded image

    predicted_class = predict_image_base64(image_filter, model)
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5051)