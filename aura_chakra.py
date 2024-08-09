import numpy as np
import cv2
import tensorflow as tf
from collections import deque
import time

# Chargement du modèle TensorFlow Lite
emotion_interpreter = tf.lite.Interpreter(model_path='model_optimized.tflite')
emotion_interpreter.allocate_tensors()
emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()

# Dictionnaire des émotions et couleurs
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emotion_colors = {
    "Angry": (0, 0, 255), "Disgusted": (0, 255, 0), "Fearful": (255, 0, 0),
    "Happy": (0, 255, 255), "Neutral": (255, 128, 0), "Sad": (255, 0, 255),
    "Surprised": (255, 255, 0)
}

class EmotionStabilizer:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.emotion_queue = deque(maxlen=window_size)

    def stabilize(self, emotions):
        self.emotion_queue.append(emotions)
        return {e: np.mean([d[e] for d in self.emotion_queue]) for e in emotions}

def optimize_face_detection(gray, scaleFactor=1.3, minNeighbors=5):
    facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return facecasc.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

def generate_auras(frame, x, y, w, h, emotions, frame_count):
    mask = np.zeros(frame.shape[:2], dtype=np.float32)
    
    # Réduire la taille des auras
    aura_size = (int(w * 1.1), int(h * 1.1))
    center = (x + w // 2, y + h // 2)
    
    # Créer un masque pour exclure le visage
    face_mask = np.ones(frame.shape[:2], dtype=np.float32)
    cv2.ellipse(face_mask, center, (w//2, h//2), 0, 0, 360, 0, -1)
    face_mask = cv2.GaussianBlur(face_mask, (9, 9), 5)

    # Aura émotionnelle
    emotion_color = np.array(emotion_colors[max(emotions, key=emotions.get)], dtype=np.float32)
    cv2.ellipse(mask, center, aura_size, 0, 0, 360, 1, -1)
    emotion_mask = cv2.GaussianBlur(mask, (15, 15), 0)
    emotion_aura = np.outer(emotion_mask, emotion_color).reshape(frame.shape)

    # Aura éthérique
    etheric_intensity = (np.sin(frame_count * 0.1) + 1) / 2
    etheric_color = np.array([100, 200, 255], dtype=np.float32)
    etheric_mask = cv2.GaussianBlur(mask, (21, 21), 0)
    etheric_aura = np.outer(etheric_mask, etheric_color).reshape(frame.shape) * etheric_intensity

    # Aura spirituelle
    spiritual_color = np.array([
        128 + 127 * np.sin(frame_count * 0.05),
        128 + 127 * np.sin(frame_count * 0.07),
        128 + 127 * np.sin(frame_count * 0.11)
    ], dtype=np.float32)
    spiritual_mask = cv2.GaussianBlur(mask, (27, 27), 0)
    spiritual_aura = np.outer(spiritual_mask, spiritual_color).reshape(frame.shape)

    # Combiner les auras
    combined_aura = (emotion_aura * 0.5 + etheric_aura * 0.3 + spiritual_aura * 0.2)
    
    # Appliquer le masque du visage
    combined_aura *= face_mask[:,:,np.newaxis]

    # Normaliser et convertir en uint8
    combined_aura = np.clip(combined_aura, 0, 255).astype(np.uint8)

    # Fusionner avec l'image originale
    result = cv2.addWeighted(frame, 1, combined_aura, 0.6, 0)
    return result

def process_face(face):
    face = cv2.resize(face, (200, 200))  # Redimensionner à 200x200
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = np.expand_dims(face, axis=0)
    face = face.astype(np.float32) / 255.0
    
    emotion_interpreter.set_tensor(emotion_input_details[0]['index'], face)
    emotion_interpreter.invoke()
    emotion_probs = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])[0]

    return {emotion_dict[i]: prob * 100 for i, prob in enumerate(emotion_probs)}

def display_global_emotions(frame, global_emotions):
    sorted_emotions = sorted(global_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
    
    y_offset = 30
    for emotion, score in sorted_emotions:
        text = f"{emotion}: {score:.1f}%"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_colors[emotion], 2)
        y_offset += 30

def main():
    cap = cv2.VideoCapture(0)
    emotion_stabilizer = EmotionStabilizer()
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = optimize_face_detection(gray)

        global_emotions = {emotion: 0 for emotion in emotion_dict.values()}
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            emotions = process_face(face)
            stabilized_emotions = emotion_stabilizer.stabilize(emotions)
            
            frame = generate_auras(frame, x, y, w, h, stabilized_emotions, frame_count)

            for emotion, score in stabilized_emotions.items():
                global_emotions[emotion] += score

        if len(faces) > 0:
            for emotion in global_emotions:
                global_emotions[emotion] /= len(faces)

            display_global_emotions(frame, global_emotions)

        cv2.imshow('Aura Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_count % 30 == 0:
            fps = frame_count / (time.time() - start_time)
            print(f"FPS: {fps:.2f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()