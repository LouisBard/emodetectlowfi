import numpy as np
import cv2
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter

# Charger le modèle TensorFlow Lite pour la reconnaissance des émotions
emotion_interpreter = tf.lite.Interpreter(model_path='model_optimized.tflite')
emotion_interpreter.allocate_tensors()
emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()

# Dictionnaire des émotions
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


# Dictionnaire des couleurs associées aux émotions
emotion_colors = {
    "Angry": (0, 0, 255),
    "Disgusted": (0, 255, 0),
    "Fearful": (255, 0, 0),
    "Happy": (0, 255, 255),
    "Neutral": (128, 128, 128),
    "Sad": (255, 0, 255),
    "Surprised": (255, 255, 0)
}



# Fonction pour générer un halo coloré autour du visage
def generate_halo(frame, x, y, w, h, emotion, radius=80):
    """
    Génère un halo coloré autour de la tête d'une personne en fonction de son émotion détectée.
    """
    # Extraire la sous-image du visage
    face = frame[y:y+h, x:x+w]

    # Redimensionner le visage pour correspondre à l'entrée du modèle
    face = cv2.resize(face, (emotion_input_details[0]['shape'][2], emotion_input_details[0]['shape'][1]), interpolation=cv2.INTER_AREA)
    face = np.expand_dims(face, axis=0)
    face = face.astype(np.float32) / 255.0

    # Appliquer la classification des émotions
    emotion_interpreter.set_tensor(emotion_input_details[0]['index'], face)
    emotion_interpreter.invoke()
    output_data = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])
    maxindex = int(np.argmax(output_data))
    emotion = emotion_dict[maxindex]

    # Générer un gradient de couleur basé sur l'émotion
    color1 = np.array(emotion_colors[emotion])
    color2 = np.array([0, 0, 0])
    halo = np.zeros((radius*2, radius*2, 3), dtype=np.uint8)
    for i in range(radius*2):
        for j in range(radius*2):
            dist = np.sqrt((i-radius)**2 + (j-radius)**2)
            if dist < radius:
                alpha = (radius - dist) / radius
                halo[i, j] = alpha * color1 + (1 - alpha) * color2

    # Stabiliser le halo pour réduire le bruit
    halo = cv2.GaussianBlur(halo, (5, 5), 0)

    # Ajouter le halo à l'image
    mask = np.zeros_like(frame)
    halo_resized = cv2.resize(halo, (w, h), interpolation=cv2.INTER_AREA)
    mask[y:y+h, x:x+w] = halo_resized
    frame = cv2.addWeighted(frame, 1.0, mask, 0.5, 0)

    return frame

# Fonction pour afficher les 3 principales émotions avec leur pourcentage
def display_emotion_info(frame, emotions):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1

    # Trier les émotions par ordre décroissant
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

    # Afficher les 3 principales émotions avec leur pourcentage
    x, y = 10, 30
    for i, (emotion, percentage) in enumerate(sorted_emotions[:3]):
        color = emotion_colors[emotion]
        text = f"{emotion}: {percentage:.2f}%"
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += 30

    return frame

# Démarrer le flux de la webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    emotions = {}
    for (x, y, w, h) in faces:
        # Ajouter le halo autour du visage
        frame = generate_halo(frame, x, y, w, h, "Happy")

        # Classifier les émotions
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (emotion_input_details[0]['shape'][2], emotion_input_details[0]['shape'][1]), interpolation=cv2.INTER_AREA)
        face = np.expand_dims(face, axis=0)
        face = face.astype(np.float32) / 255.0
        emotion_interpreter.set_tensor(emotion_input_details[0]['index'], face)
        emotion_interpreter.invoke()
        output_data = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])
        emotion_probs = output_data[0]

        # Mettre à jour le dictionnaire des émotions
        for i, prob in enumerate(emotion_probs):
            emotion = emotion_dict[i]
            emotions[emotion] = prob * 100

    # Afficher les 3 principales émotions avec leur pourcentage
    frame = display_emotion_info(frame, emotions)

    # Afficher l'image avec l'effet appliqué
    cv2.imshow('Video', cv2.resize(frame, (1024, 768), interpolation=cv2.INTER_LANCZOS4))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()