import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter
from collections import deque
import time

# Charger le modèle TensorFlow Lite pour la reconnaissance des émotions
emotion_interpreter = Interpreter(model_path='model_optimized.tflite')
emotion_interpreter.allocate_tensors()
emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()

# Dictionnaire des émotions
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Dictionnaire des couleurs associées aux émotions (modifié pour BGR)
emotion_colors = {
    "Angry": (0, 0, 255),      # Rouge
    "Disgusted": (0, 255, 0),  # Vert
    "Fearful": (128, 0, 128),  # Violet
    "Happy": (0, 255, 255),    # Jaune
    "Neutral": (192, 192, 192),# Gris clair
    "Sad": (255, 0, 0),        # Bleu
    "Surprised": (0, 165, 255) # Orange
}

# Dictionnaire des poids des émotions
emotion_weights = {
    "Happy": 3,
    "Sad": -2,
    "Angry": -3,
    "Fearful": -2,
    "Surprised": 1,
    "Disgusted": -2,
    "Neutral": 0
}

class EmotionStabilizer:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.emotion_queue = deque(maxlen=window_size)

    def stabilize(self, emotions):
        self.emotion_queue.append(emotions)
        avg_emotions = {}
        for emotion in emotion_dict.values():
            avg_emotions[emotion] = np.mean([e[emotion] for e in self.emotion_queue])
        return avg_emotions

class RadiusStabilizer:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.radius_queue = deque(maxlen=window_size)

    def stabilize(self, radius):
        self.radius_queue.append(radius)
        return int(np.mean(self.radius_queue))

class DominantEmotionStabilizer:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.emotion_queue = deque(maxlen=window_size)

    def stabilize(self, emotion):
        self.emotion_queue.append(emotion)
        emotion_counts = {}
        for e in self.emotion_queue:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        return max(emotion_counts, key=emotion_counts.get)

def optimize_face_detection(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)):
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    return facecasc.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)

def generate_improved_emotion_aura(frame, center, dominant_emotion, radius, time_factor):
    height, width = frame.shape[:2]
    y, x = np.ogrid[:height, :width]
    
    aspect_ratio = 1.5
    distances = np.sqrt((x - center[0])**2 + ((y - center[1])/aspect_ratio)**2)
    
    pulse_factor = 0.05 * np.sin(time_factor * np.pi) + 1
    normalized_distances = distances / (radius * pulse_factor)
    
    emotional_color = np.array(emotion_colors[dominant_emotion]) / 255.0
    etheric_color = np.array([1.0, 1.0, 1.0])  # White
    mental_color = np.array([0.54, 0.17, 0.89])  # Blue violet

    base_aura = np.exp(-normalized_distances**2 / 0.5)
    color_variation = np.sin(normalized_distances * 10 + time_factor) * 0.5 + 0.5

    aura = (etheric_color * np.exp(-normalized_distances/0.2)[..., np.newaxis] +
            emotional_color * np.exp(-(normalized_distances-0.3)**2/0.1)[..., np.newaxis] * color_variation[..., np.newaxis] +
            mental_color * np.exp(-(normalized_distances-0.6)**2/0.2)[..., np.newaxis] * (1 - color_variation[..., np.newaxis])) * base_aura[..., np.newaxis]

    pulse = np.sin(time_factor * 2) * 0.1 + 0.9
    aura *= pulse

    aura = np.clip(aura, 0, 1)
    aura = (aura * 255).astype(np.uint8)

    aura = cv2.GaussianBlur(aura, (15, 15), 0)

    result = cv2.addWeighted(frame, 0.7, aura, 0.8, 0)

    return result

def calculate_emotion_score(emotions):
    total_score = sum((percentage / 100) * emotion_weights.get(emotion, 0) for emotion, percentage in emotions.items())
    
    min_possible_score = sum(weight for weight in emotion_weights.values() if weight < 0)
    max_possible_score = sum(weight for weight in emotion_weights.values() if weight > 0)
    
    normalized_score = (total_score - min_possible_score) / (max_possible_score - min_possible_score) * 10
    
    return max(0, min(10, normalized_score))

def optimize_display_emotion_info(frame, emotions):
    font = cv2.FONT_HERSHEY_SIMPLEX
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]

    padding = 5
    box_width = 180
    line_height = 20
    box_height = (len(sorted_emotions) + 1) * line_height + 2 * padding
    start_x, start_y = 10, 10

    cv2.rectangle(frame, (start_x, start_y), (start_x + box_width, start_y + box_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (start_x, start_y), (start_x + box_width, start_y + box_height), (255, 255, 255), 1)

    cv2.putText(frame, "Overall emotions", (start_x + padding, start_y + line_height), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for i, (emotion, percentage) in enumerate(sorted_emotions):
        cv2.putText(frame, f"{emotion}: {percentage:.1f}%", (start_x + padding, start_y + (i + 2) * line_height), font, 0.4, emotion_colors[emotion], 1, cv2.LINE_AA)

    return frame

def display_compact_stylized_ed_badge(frame, x, y, w, h, score):
    badge_width, badge_height = 60, 30
    badge_x, badge_y = x + w + 5, y

    frame_height, frame_width = frame.shape[:2]
    if badge_x + badge_width > frame_width:
        badge_x = x - badge_width - 5
    if badge_y + badge_height > frame_height:
        badge_y = y + h - badge_height

    base_color = (0, 200, 0) if score >= 7 else (0, 200, 200) if score >= 4 else (0, 0, 200)

    overlay = frame.copy()
    cv2.rectangle(overlay, (badge_x, badge_y), (badge_x + badge_width, badge_y + badge_height), base_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_width, badge_y + badge_height), base_color, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"{score:.1f}", (badge_x + 5, badge_y + badge_height - 8), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "E.D.", (badge_x + badge_width - 25, badge_y + 12), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return frame

def process_face(face, input_width, input_height):
    face = cv2.resize(face, (input_width, input_height))
    face = np.expand_dims(face.astype(np.float32) / 255.0, axis=0)
    
    emotion_interpreter.set_tensor(emotion_input_details[0]['index'], face)
    emotion_interpreter.invoke()
    emotion_probs = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])[0]

    return {emotion_dict[i]: prob * 100 for i, prob in enumerate(emotion_probs)}

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    emotion_stabilizer = EmotionStabilizer()
    radius_stabilizer = RadiusStabilizer()
    dominant_emotion_stabilizer = DominantEmotionStabilizer()
    input_shape = emotion_input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]

    start_time = time.time()
    frame_count = 0
    last_fps_time = start_time

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read webcam frame.")
            break

        frame_count += 1
        current_time = time.time()
        if current_time - last_fps_time >= 1:
            fps = frame_count / (current_time - last_fps_time)
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            last_fps_time = current_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = optimize_face_detection(gray)

        time_factor = (current_time - start_time) % 2

        stabilized_emotions = {}  # Initialiser stabilized_emotions ici

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            emotions = process_face(face, input_width, input_height)
            stabilized_emotions = emotion_stabilizer.stabilize(emotions)

            center = (x + w // 2, y + h // 2)
            raw_radius = max(w, h) * 1.1
            stable_radius = radius_stabilizer.stabilize(raw_radius)
            
            raw_dominant_emotion = max(emotions, key=emotions.get)
            stable_dominant_emotion = dominant_emotion_stabilizer.stabilize(raw_dominant_emotion)
            frame = generate_improved_emotion_aura(frame, center, stable_dominant_emotion, stable_radius, time_factor)

            score = calculate_emotion_score(emotions)
            frame = display_compact_stylized_ed_badge(frame, x, y, w, h, score)

        if stabilized_emotions:  # Vérifier si des émotions ont été détectées
            frame = optimize_display_emotion_info(frame, stabilized_emotions)

        cv2.imshow('Emotion Detection', cv2.resize(frame, (800, 600), interpolation=cv2.INTER_AREA))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()