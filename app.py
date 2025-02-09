from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import os
import uuid  # For generating unique filenames
import logging
import train_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Model Paths
ANN_MODEL_PATH = 'model/ann_model.h5'
CNN_MODEL_PATH = 'model/cnn_model.h5'
STATIC_DIR = 'static'

# Ensure 'static' directory exists
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
    logging.info(f"Created directory: {STATIC_DIR}")

# Function to train and save models
def train_and_save_models():
    logging.info("Training Models...")
    # Load and preprocess data
    try:
        x_train_ann, x_val_ann, x_test_ann, train_generator, x_val_cnn, x_test_cnn, y_train, y_val, y_test = train_model.load_and_preprocess_data()
    except Exception as e:
        logging.exception(f"Error loading and preprocessing data: {e}")
        return False  # Indicate failure

    # Train ANN model
    try:
        ann_model = train_model.build_model(model_type="ANN", hidden_layers_neurons=[256, 128], dropout_rate=0.2, l2_lambda=0.001)
        ann_history, ann_trained_model = train_model.train_model(ann_model, x_train_ann, y_train, x_val_ann, y_val, epochs=50, batch_size=64, patience=5, learning_rate=0.001)  # Increased epochs
        ann_trained_model.save(ANN_MODEL_PATH)
        logging.info("ANN Model Trained and Saved.")
    except Exception as e:
        logging.exception(f"Error training and saving ANN model: {e}")
        return False

    # Train CNN model
    try:
        cnn_model = train_model.build_model(model_type="CNN", dropout_rate=0.2, l2_lambda=0.001)
        cnn_history, cnn_trained_model = train_model.train_model(cnn_model, train_generator, y_train, x_val_cnn, y_val, epochs=50, batch_size=64, patience=5, learning_rate=0.001, use_generator=True)  # Increased epochs
        cnn_trained_model.save(CNN_MODEL_PATH)
        logging.info("CNN Model Trained and Saved.")
    except Exception as e:
        logging.exception(f"Error training and saving CNN model: {e}")
        return False

    return True  # Indicate success

# Train and save the model if it doesn't exist
if not os.path.exists(ANN_MODEL_PATH) or not os.path.exists(CNN_MODEL_PATH):
    from sklearn.model_selection import train_test_split
    if not train_and_save_models():
        logging.error("Failed to train and save models during startup.  Application may not function correctly.")
        # Optionally, exit the application if model training is critical

# Load the models
try:
    ann_model = tf.keras.models.load_model(ANN_MODEL_PATH)
    logging.info("ANN model loaded successfully.")
except Exception as e:
    logging.exception(f"Error loading ANN model: {e}")
    ann_model = None  # Handle the case where the model fails to load

try:
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    logging.info("CNN model loaded successfully.")
except Exception as e:
    logging.exception(f"Error loading CNN model: {e}")
    cnn_model = None # Handle the case where the model fails to load

def preprocess_image(image_path):
    """Preprocesses the image for digit recognition."""
    try:
        img = Image.open(image_path).convert('L')  # Grayscale

        # Noise Reduction (BEFORE thresholding)
        img = img.filter(ImageFilter.MedianFilter(size=3))

        # Otsu's Thresholding
        threshold = get_otsu_threshold(np.array(img))
        img = img.point(lambda x: 0 if x < threshold else 255, '1')  # Binary

        # Robust Inversion Decision: Count pixels and invert if necessary
        dark_pixels = np.sum(np.array(img) == 0)
        light_pixels = np.sum(np.array(img) == 255)

        # Invert the image only if there are significantly more light pixels than dark pixels
        #NO inversion

        # Resize and Center the Digit
        img = img.resize((40, 40), Image.LANCZOS) #resize image
        background = Image.new('L', (40, 40), 0) #create black background
        bbox = img.getbbox()

        if bbox:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            x = (40 - width) // 2
            y = (40 - height) // 2
            cropped_img = img.crop(bbox)
            background.paste(cropped_img, (x, y))

        img = background.resize((28, 28), Image.LANCZOS)

        # Normalize Pixel Values
        img_array = np.array(img).astype('float32') / 255.0

        return img_array

    except Exception as e:
        logging.exception(f"Error preprocessing image {image_path}: {e}")
        return None

def get_otsu_threshold(image_array):
    """Calculates Otsu's threshold for a grayscale image."""
    # Flatten the image array
    pixels = image_array.flatten()
    
    # Calculate histogram
    histogram, bin_edges = np.histogram(pixels, bins=256, range=(0, 256))
    
    total_pixels = len(pixels)
    
    # Initialize variables
    max_variance = 0
    threshold = 0
    
    sum_background = 0
    weight_background = 0
    
    # Iterate over all possible threshold values (0 to 255)
    for t in range(256):
        # Update background and foreground statistics
        weight_background += histogram[t]
        
        if weight_background == 0:
            continue
        
        weight_foreground = total_pixels - weight_background
        
        if weight_foreground == 0:
            break
        
        sum_background += t * histogram[t]
        
        # Calculate means
        mean_background = sum_background / weight_background
        mean_foreground = (np.sum(np.arange(t + 1, 256) * histogram[t + 1:])) / weight_foreground
        
        # Calculate between-class variance
        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        # Update threshold if variance is larger
        if between_class_variance > max_variance:
            max_variance = between_class_variance
            threshold = t
    
    return threshold

# Prediction functions
def predict_ann(image_path):
    """Predicts the digit using the ANN model."""
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return None, None  # Return None if preprocessing failed
    processed_image = processed_image.reshape(1, 784)  # Flatten for ANN
    try:
        prediction = ann_model.predict(processed_image)
        predicted_digit = np.argmax(prediction)
        confidence = prediction[0][predicted_digit]
        return predicted_digit, confidence
    except Exception as e:
        logging.exception(f"Error predicting with ANN model: {e}")
        return None, None  # Return None if prediction fails

def predict_cnn(image_path):
    """Predicts the digit using the CNN model."""
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return None, None  # Return None if preprocessing failed

    processed_image = processed_image.reshape(1, 28, 28, 1)  # Reshape for CNN
    try:
        prediction = cnn_model.predict(processed_image)
        predicted_digit = np.argmax(prediction)
        confidence = prediction[0][predicted_digit]
        return predicted_digit, confidence
    except Exception as e:
        logging.exception(f"Error predicting with CNN model: {e}")
        return None, None # Return None if prediction fails

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'image' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['image']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        try:
            if file:
                # Generate a unique filename
                filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1] # Ensure unique filename
                file_path = os.path.join(STATIC_DIR, filename)
                file.save(file_path)
                logging.info(f"File saved to: {file_path}")

                # Determine which model to use
                model_type = request.form.get('model_type', 'cnn')  # Get selected model type
                if model_type not in ['ann', 'cnn']:
                    model_type = 'cnn'  # Default to CNN if invalid

                # Make prediction
                if model_type == 'ann':
                    if ann_model is None:
                        return render_template('index.html', error='ANN model not loaded.')
                    predicted_digit, confidence = predict_ann(file_path)
                else:  # Default to CNN
                    if cnn_model is None:
                        return render_template('index.html', error='CNN model not loaded.')
                    predicted_digit, confidence = predict_cnn(file_path)

                if predicted_digit is not None:
                     return render_template('result.html', predicted_digit=predicted_digit, confidence=confidence, image_path=filename, model_type=model_type) # just pass filename

                else:
                    return render_template('index.html', error='Prediction failed.')

        except Exception as e:
            logging.exception("An error occurred during file processing or prediction.") # Log the full exception
            return render_template('index.html', error=f'An error occurred: {e}') # Show a user-friendly error
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)