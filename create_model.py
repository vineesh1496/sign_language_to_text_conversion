import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

def create_model():
    model = Sequential([
        Input(shape=(128, 128, 1)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(96, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(27, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    model = create_model()
    model.summary()
    
    # Save the model architecture and weights
    model_json = model.to_json()
    with open("Models/model_new_fixed.json", "w") as json_file:
        json_file.write(model_json)
    
    # Create dummy weights and save
    model.save_weights("Models/model_new_fixed.weights.h5")
    print("Fixed model saved successfully!")
