import os
import io
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import predict

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Load ONNX model
model_session = predict.load_model()

@app.route('/')
def index():
    """Render the homepage with upload form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def prediction():
    """Process image and return prediction results"""
    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    # If no file was selected
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read image file
        img_bytes = file.read()
        
        # Save the image for display
        img = Image.open(io.BytesIO(img_bytes))
        
        # Resize for display if needed
        display_img = img.copy()
        if display_img.mode != 'L':
            display_img = display_img.convert('L')
        
        # Generate a secure filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        display_img.save(filepath)
        
        # Make prediction
        digit, probabilities = predict.predict(img_bytes, model_session)
        
        if digit is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Calculate confidence percentage
        confidence = int(round(probabilities[digit] * 100))
        
        # Return prediction results
        return jsonify({
            'success': True,
            'digit': int(digit),
            'confidence': confidence,
            'probabilities': probabilities,
            'image_path': filepath
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/static/samples/<filename>')
def sample_file(filename):
    """Serve sample MNIST digits"""
    return send_from_directory('static/samples', filename)

if __name__ == '__main__':
    app.run(debug=True)
