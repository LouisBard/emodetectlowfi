import numpy as np
import argparse
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

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

# command line argument
#ap = argparse.ArgumentParser()
#ap.add_argument("--mode",help="train/display")
#mode = ap.parse_args().mode
mode = "display"

# Load model transfer style
style_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Apply style
def apply_style(content_image, style_image):
    content_image = tf.image.convert_image_dtype(content_image, tf.float32)
    style_image = tf.image.convert_image_dtype(style_image, tf.float32)
    content_image = tf.image.resize(content_image, (256, 256))
    style_image = tf.image.resize(style_image, (256, 256))
    outputs = style_model(tf.constant(content_image[tf.newaxis, ...]), tf.constant(style_image[tf.newaxis, ...]))
    stylized_image = outputs[0]
    return np.array(stylized_image[0])

# Définir les chemins des images de style pour chaque émotion

style_images = {
    'Happy': r'./images/happy.jpg',
    'Sad': r'./images/sad.jpeg',
    'Angry': r'./images/angry.jpeg',
    'Surprised': r'./images/surprised.jpeg',
    'Fearful': r'./images/fear.jpeg',
    'Disgusted': r'./images/disgusted.jpg',
    'Neutral': r'./images/disgusted.jpg'
}

default_style_image_path = r'./images/angry.jpeg'


def load_img(path):
    if path is None or not tf.io.gfile.exists(path):
        path = default_style_image_path
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (256, 256))
    return img

def load_img(path):
    try:
        # Vérifier que le chemin est valide
        if path is None or not os.path.exists(path):
            print(f"Chemin invalide ou fichier non trouvé: {path}")
            path = default_style_image_path

        # Charger l'image en utilisant OpenCV pour vérifier les erreurs
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Erreur lors du chargement de l'image: {path}")

        # Convertir l'image en format TensorFlow
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        img = tf.image.resize(img, (256, 256))
        img = img / 255.0  # Normaliser l'image

        return img
    except Exception as e:
        print(f"Erreur lors du chargement de l'image: {e}")
        raise

# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

# Define data generators
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 50

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=(48,48),
#         batch_size=batch_size,
#         color_mode="grayscale",
#         class_mode='categorical')

# validation_generator = val_datagen.flow_from_directory(
#         val_dir,
#         target_size=(48,48),
#         batch_size=batch_size,
#         color_mode="grayscale",
#         class_mode='categorical')

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

# If you want to train the same model or try other models, go for this
if mode == "train":
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
    model_info = model.fit_generator(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size)
    plot_model_history(model_info)
    model.save_weights('model.h5')

# emotions will be displayed on your face from the webcam feed
elif mode == "display":
    model.load_weights('model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

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

        style_image = load_img(default_style_image_path)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            style_image_path = style_images[emotion_dict[maxindex]]
            print(style_image_path)
            style_image = load_img(style_image_path)
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Appliquer le style transfer
        stylized_frame = apply_style(frame, style_image)
      #  cv2.imshow(stylized_frame)
        # Resize
        cv2.imshow('Video', cv2.resize(stylized_frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
