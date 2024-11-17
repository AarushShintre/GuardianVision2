import tensorflow as tf
import numpy as np

from constants import BEHAVIORS

# Load the saved TensorFlow model
model_path = "path/to/your/saved/model"  # Replace with your model's path
model = tf.keras.models.load_model(model_path)

# Incoming data to classify (example)
# Replace `incoming_data` with your preprocessed input data
incoming_data = np.array([
    # Example data - this should match the input shape of the model
    # Replace this with actual data to classify
    [0.1, 0.2, 0.3, 0.4, 0.5], 
    [0.6, 0.7, 0.8, 0.9, 1.0]
])

# Ensure data matches the input shape of the model
# Adjust as needed depending on your data preprocessing
incoming_data = np.expand_dims(incoming_data, axis=0) if len(incoming_data.shape) == 1 else incoming_data

# Make predictions
predictions = model.predict(incoming_data)

# Map predictions to behavior classes
classified_behaviors = [BEHAVIORS[np.argmax(pred)] for pred in predictions]

# Output the results
for i, behavior in enumerate(classified_behaviors):
    print(f"Data Point {i + 1}: Classified as {behavior}")
