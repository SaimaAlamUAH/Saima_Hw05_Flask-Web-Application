import numpy as np
import onnxruntime as ort
from PIL import Image
import io

def load_model(model_path="mnist_cnn_model.onnx"):
    """Load the ONNX model for inference"""
    try:
        session = ort.InferenceSession(model_path)
        return session
    except Exception as e:
        print(f"Error loading the ONNX model: {e}")
        return None

def preprocess_image(image_bytes):
    """Preprocess image for model inference"""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize((28, 28))
    img_array = np.array(image)
    # Invert if background is white (MNIST expects white digits on black)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 1, 28, 28).astype(np.float32)
    return img_array

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predict(image_bytes, model_session=None):
    """Make digit prediction using ONNX model"""
    if model_session is None:
        model_session = load_model()
    if model_session is None:
        return None, None
    img_array = preprocess_image(image_bytes)
    input_name = model_session.get_inputs()[0].name
    output_name = model_session.get_outputs()[0].name
    results = model_session.run([output_name], {input_name: img_array})
    logits = results[0][0]  # These are raw scores (logits)
    probabilities = softmax(logits)  # Now these sum to 1 and are in [0, 1]
    predicted_class = int(np.argmax(probabilities))
    return predicted_class, probabilities.tolist()
