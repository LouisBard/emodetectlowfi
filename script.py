import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Enable Metal plugin
tf.config.experimental.set_visible_devices(
    tf.config.list_physical_devices('GPU'), 'GPU'
)

# Enable mixed precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Verify GPU usage
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Is GPU available: ", tf.test.is_gpu_available())

mode = "display"

# Load model transfer style
style_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Apply style
def apply_style(content_image, style_image):
    content_image = tf.image.convert_image_dtype(content_image, tf.float32)
    content_image = tf.image.resize(content_image, (256, 256))
    outputs = style_model(tf.constant(content_image[tf.newaxis, ...]), tf.constant(style_image[tf.newaxis, ...]))
    stylized_image = outputs[0]
    return np.array(stylized_image[0])

# Define paths for style images
style_images = {
    'Happy': './images/happy',
    'Sad': './images/sad',
    'Angry': './images/angry',
    'Surprised': './images/surprised',
    'Fearful': './images/fear',
    'Disgusted': './images/disgusted',
    'Neutral': './images/happy'
}

default_style_image_path = './images/angry'

def load_and_preprocess_img(path):
    try:
        # Find the file regardless of extension
        files = glob.glob(f"{path}.*")
        if not files:
            print(f"No file found for: {path}")
            files = glob.glob(f"{default_style_image_path}.*")

        if not files:
            raise ValueError(f"No default image found at: {default_style_image_path}")

        img_path = files[0]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Error loading image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        img = tf.image.resize(img, (128, 128))
        img = img / 255.0  # Normalize the image

        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        raise

# Preload and preprocess style images
preloaded_style_images = {emotion: load_and_preprocess_img(path) for emotion, path in style_images.items()}

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Load the model weights
model.load_weights('model.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Sharpening kernel
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

# start the webcam feed
cap = cv2.VideoCapture(0)
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    style_image = preloaded_style_images['Angry']  # Default style

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        emotion = emotion_dict[maxindex]
        style_image = preloaded_style_images[emotion]
        cv2.putText(frame, emotion, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    frame = apply_style(frame, style_image)

    # Convert frame to uint8
    frame = (frame * 255).astype(np.uint8)

    # Apply sharpening
    frame = cv2.filter2D(frame, -1, sharpen_kernel)

    # Resize and display
    cv2.imshow('Video', cv2.resize(frame,(1024,768),interpolation = cv2.INTER_LANCZOS4))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
