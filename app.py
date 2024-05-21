# app.py
from flask import Flask, request, jsonify, render_template
from torchvision import transforms
from PIL import Image
import torch
import timm

app = Flask(__name__, static_url_path='/static')

# Load the state dictionary of the trained model
model_state_dict = torch.load('fine_tuned_vit.pth', map_location=torch.device('cpu'))

# Define and load the ViT model using timm
model = timm.create_model('vit_base_patch32_224', pretrained=False, num_classes=10)
model.load_state_dict(model_state_dict)
model.eval()

# Define image preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define class names (replace with your actual class names)
class_names = ['Bacterial spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 'Septoria leaf spot', 
               'Two-spotted spider mite', 'Target Spot', 'Yellow Leaf Curl Virus', 'Mosaic virus', 'Healthy']

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/LM')
def LM():
    return render_template('Diseases/LM.html')

@app.route('/BS')
def BS():
    return render_template('Diseases/BS.html')

@app.route('/MV')
def MV():
    return render_template('Diseases/MV.html')

@app.route('/YLCV')
def YLCV():
    return render_template('Diseases/YLCV.html')

@app.route('/TS')
def TS():
    return render_template('Diseases/TS.html')

@app.route('/TSSM')
def TSSM():
    return render_template('Diseases/TSSM.html')

@app.route('/LB')
def LB():
    return render_template('Diseases/LB.html')

@app.route('/SLS')
def SLS():
    return render_template('Diseases/SLS.html')

@app.route('/EB')
def EB():
    return render_template('Diseases/EB.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # Check if the file is not empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file is an allowed format (you can add more image formats as needed)
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            # Open and preprocess the image
            image = Image.open(file).convert('RGB')
            image = preprocess(image).unsqueeze(0)

            # Perform prediction
            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                prediction = class_names[predicted.item()]

            return jsonify({'prediction': prediction})

        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Unsupported file format'})

if __name__ == '__main__':
    app.run(debug=True)
