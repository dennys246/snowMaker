import cv2, os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Labeler:
    """ A class to handle labeling of images for snow detection using a CNN model.
    This class initializes a CNN model, loads a dataset, and provides methods for training and evaluating the model.
    """

    def __init__(self):
        # Define labels
        self.x_train = None

        # Load model if it exists
        try:
            self.model = tf.keras.models.load_model('label_model.h5')
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.build()
        
    def build(self):
        # Define a simple CNN model
        self.model = tf.keras.Sequential([
            # First convolutional layer
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),

            # Second convolutional layer
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),

            # Single density layer
            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),

            # Output layer
            tf.keras.layers.Dense(10, activation='softmax')  # Assuming 10 classes for labeling
        ])
        
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        print("Label model built successfully.")

    def load_dataset(self):
        # Load dataset (train and test)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()

        # Reshape the data to fit the model input
        print("Original x_train shape:", self.x_train.shape)
        print("Original x_test shape:", self.x_test.shape)
        self.x_train = self.x_train.reshape((self.x_train.shape[0], 28, 28, 1))
        self.x_test = self.x_test.reshape((self.x_test.shape[0], 28, 28, 1))

        # Normalize the images to [0, 1]
        self.x_train = self.x_train.astype("float32") / 255.0
        self.x_test = self.x_test.astype("float32") / 255.0

    def train(self):
        """
        Train the CNN model on the loaded dataset.
        """
        # Check if the dataset is loaded, if not, load it
        if self.x_train is None:
            self.load_dataset()

        # Train the model and capture history
        history = self.model.fit(self.x_train, self.y_train, epochs=5, validation_data=(self.x_test, self.y_test))

        #Plot the training history
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.show()
        # Save the training history to a file
        with open('training_history.txt', 'w') as f:
            for key in history.history.keys():
                f.write(f"{key}: {history.history[key]}\n")
    

        # Save the model
        self.model.save('label_model.h5')
        print("Model trained and saved successfully.")

    def evaluate(self):
        """
        Evaluate the CNN model on the test dataset.
        """
        # Check if the dataset is loaded, if not, load it
        if self.x_test is None:
            self.load_dataset()

        # Evaluate the model
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
        return loss, accuracy

    def predict(self, image):
        """
        Predict the label of a single image using the trained CNN model.
        
        Args:
            image (np.ndarray): The input image to be classified.
        
        Returns:
            int: The predicted label for the input image.
        """
        # Preprocess the image
        image = cv2.resize(image, (28, 28))  # Resize to match model input
        image = image.astype("float32") / 255.0  # Normalize the image
        image = np.reshape(image, (1, 28, 28, 1)) # Add batch dimension
        
        # Predict the label
        prediction = self.model.predict(image)
        predicted_label = np.argmax(prediction, axis=1)[0]
        print(f"Predicted label: {predicted_label} with probability {np.max(prediction)}")
        return predicted_label
    
    def load_model(self, model_path = 'label_model.h5'):
        """
        Load a pre-trained model from the specified path.
        
        Args:
            model_path (str): Path to the pre-trained model file.
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.build()

if __name__ == "__main__":
    labeler = Labeler()
    # Uncomment below lines to train the model
    labeler.load_dataset()
    labeler.train()