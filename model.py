import tensorflow as tf

def load_model():
    # Load your pre-trained model here
    model = tf.keras.models.load_model('path_to_your_model.h5')  # Replace with your model's path
    return model