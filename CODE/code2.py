"""
This is File is Created by Bhuvanesh. Y 23BFN0037 and this lines are added by me.
And this code runs on WSL2 for CUDA support.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("GPU Available:", tf.config.list_physical_devices('GPU'), "\n\n")

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 4
EPOCHS = 50  # Increased max epochs since we'll use early stopping


def prepare_dataset():
    """
    Check if dataset exists, otherwise create a minimal test dataset
    """
    # Create directories if they don't exist and change the path if you are using windows
    os.makedirs("/mnt/d/DA's/DAs-Ai_and_Python/Source Files/Image Recognition/Datasets/Train/pen", exist_ok=True)
    os.makedirs("/mnt/d/DA's/DAs-Ai_and_Python/Source Files/Image Recognition/Datasets/Train/pencil", exist_ok=True)

    os.makedirs("/mnt/d/DA's/DAs-Ai_and_Python/Source Files/Image Recognition/Datasets/Test/pen", exist_ok=True)
    os.makedirs("/mnt/d/DA's/DAs-Ai_and_Python/Source Files/Image Recognition/Datasets/Test/pencil", exist_ok=True)

    # Check if we have enough images and change the path if you are using windows
    total_train_images = sum(
        [len(os.listdir(f"/mnt/d/DA's/DAs-Ai_and_Python/Source Files/Image Recognition/Datasets/Train/{cls}")) for cls
         in ['pen', 'pencil']])
    total_val_images = sum(
        [len(os.listdir(f"/mnt/d/DA's/DAs-Ai_and_Python/Source Files/Image Recognition//Datasets/Test/{cls}")) for cls
         in ['pen', 'pencil']])

    print(f"Found {total_train_images} training images and {total_val_images} validation images")

    if total_train_images < 60 or total_val_images < 20:
        print(
            "Not enough images found. Please ensure you have 30 images per class for training and 10 per class for validation.")
        return False
    return True


def setup_data_generators():
    train_data_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_data_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_data_gen.flow_from_directory(
        "/mnt/d/DA's/DAs-Ai_and_Python/Source Files/Image Recognition/Datasets/Train", # Change the path if you are using windows
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    validation_generator = validation_data_gen.flow_from_directory(
        "/mnt/d/DA's/DAs-Ai_and_Python/Source Files/Image Recognition/Datasets/Test", # Change the path if you are using windows
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )

    return train_generator, validation_generator


def build_model():
    # Build the model with Input layer explicitly defined
    model = Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def setup_callbacks():
    """
    Setup callbacks for model convergence monitoring
    """
    # Model checkpoint to save the best model
    checkpoint = ModelCheckpoint(
        "/mnt/d/DA's/DAs-Ai_and_Python/Source Files/Image Recognition/Code/best_model.h5", # Change the path if you are using windows
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # Early stopping to prevent overfitting and enable convergence-based training
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,  # Number of epochs with no improvement after which training will stop
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001  # Minimum change to qualify as improvement
    )

    # Reduce learning rate when a metric has stopped improving
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,  # Factor by which the learning rate will be reduced
        patience=5,  # Number of epochs with no improvement after which learning rate will be reduced
        min_lr=1e-6,  # Lower bound on the learning rate
        verbose=1
    )

    return [checkpoint, early_stopping, reduce_lr]


def main():
    # Prepare dataset
    dataset_ready = prepare_dataset()
    if not dataset_ready:
        return

    # Setup data generators
    train_generator, validation_generator = setup_data_generators()

    # Check if generators have data
    if train_generator.samples == 0:
        print("Error: No training data found. Please add images to the training folders.")
        return

    if validation_generator.samples == 0:
        print("Error: No validation data found. Please add images to the validation folders.")
        return

    # Calculate proper steps
    steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
    validation_steps = max(1, validation_generator.samples // BATCH_SIZE)

    print(f"Training with {train_generator.samples} images in {steps_per_epoch} steps per epoch")
    print(f"Validating with {validation_generator.samples} images in {validation_steps} steps")

    # Build the model
    model = build_model()
    model.summary()

    # Setup callbacks for convergence monitoring
    callbacks = setup_callbacks()

    # Train the model with proper error handling
    try:
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks  # Add callbacks for convergence monitoring
        )

        # Load the best model (saved by ModelCheckpoint)
        best_model = load_model("/mnt/d/DA's/DAs-Ai_and_Python/Source Files/Image Recognition/Code/best_model.h5") # Change the path if you are using windows

        # Save as the standard filename for compatibility with existing code
        best_model.save("/mnt/d/DA's/DAs-Ai_and_Python/Source Files/Image Recognition/Code/image_recognition_model.h5") # Change the path if you are using windows

        # Evaluate the model
        loss, accuracy = best_model.evaluate(validation_generator, steps=validation_steps)
        print(f'Validation Accuracy: {accuracy * 100:.2f}%')

        # Plot training history with convergence visualization
        plt.figure(figsize=(15, 10))

        # Accuracy subplot
        plt.subplot(2, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.legend()

        # Loss subplot
        plt.subplot(2, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.legend()

        # Learning rate subplot if available
        if 'lr' in history.history:
            plt.subplot(2, 2, 3)
            plt.plot(history.history['lr'])
            plt.title('Learning Rate')
            plt.yscale('log')

        # Convergence metric: delta between train and validation loss
        plt.subplot(2, 2, 4)
        delta_loss = [t - v for t, v in zip(history.history['loss'], history.history['val_loss'])]
        plt.plot(delta_loss, label='Train-Val Loss Difference')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Convergence Metric')
        plt.legend()

        plt.tight_layout()
        plt.savefig('/mnt/d/DA\'s/DAs-Ai_and_Python/Source Files/Image Recognition/Code/training_history.png')
        print("Training history saved as training_history.png")

        # Calculate and print convergence statistics
        final_epoch = len(history.history['loss'])
        stopping_epoch = final_epoch
        for i, callback in enumerate(callbacks):
            if isinstance(callback, EarlyStopping) and callback.stopped_epoch > 0:
                stopping_epoch = callback.stopped_epoch + 1
                break

        print(f"\nConvergence Statistics:")
        print(f"Total epochs trained: {final_epoch}")
        if stopping_epoch < EPOCHS:
            print(f"Early stopping activated at epoch {stopping_epoch} due to convergence")

        if 'lr' in history.history:
            print(f"Initial learning rate: {history.history['lr'][0]}")
            print(f"Final learning rate: {history.history['lr'][-1]}")

        print(f"Best validation accuracy: {max(history.history['val_accuracy']) * 100:.2f}%")
        print(f"Best validation loss: {min(history.history['val_loss']):.4f}")

    except Exception as e:
        print(f"Error during training: {e}")


def predict_image_class(image_path):

    class_labels = ['pen', 'pencil']

    # Load the trained model
    model = load_model('/mnt/d/DA\'s/DAs-Ai_and_Python/Source Files/Image Recognition/Code/image_recognition_model.h5') # Change the path if you are using windows

    # Load and preprocess the image
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    print(f"The image is predicted to be a: {predicted_class} (Confidence: {confidence:.2f}%)")
    return predicted_class


def pred():
    # Predict the class of an Collected from image 
    while True:
        tpath = input("Enter the path of the image if using wsl give wsl pathfile. (Type C to quit.) : ")
        if tpath.upper() == 'C':
            break
        elif tpath:
            print(predict_image_class(tpath))
        else:
            print("Invalid Input")
        


if __name__ == "__main__":
    inp = input("Do you want to train the model? (Y/N) : ")
    if inp.lower() == 'y':
        main()
    elif inp.lower() == 'n':
        pred()
    else:
        print("Invalid Input")