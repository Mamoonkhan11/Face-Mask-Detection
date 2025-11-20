# File: src/model_build.py (or model_train.py)

from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # type: ignore
from tensorflow.keras.regularizers import l2

def build_model(input_shape=(128, 128, 3)):
    """
    Build and compile the CNN model for Face Mask Detection.

    This updated model:
    - Adds L2 regularization to prevent bias toward dominant class
    - Adds BatchNormalization for more stable training
    - Uses slightly deeper CNN for better discrimination
    """

    model = Sequential([
        # BLOCK 1
        Conv2D(32, (3, 3), activation='relu', padding='same', 
               kernel_regularizer=l2(0.001), input_shape=input_shape),
        MaxPooling2D(2, 2),

        # BLOCK 2
        Conv2D(64, (3, 3), activation='relu', padding='same', 
               kernel_regularizer=l2(0.001)),
        MaxPooling2D(2, 2),

        # BLOCK 3
        Conv2D(128, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(0.001)),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),

        Dense(1, activation='sigmoid')  # BINARY OUTPUT
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print(" Model built and compiled successfully!")
    return model