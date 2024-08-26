import numpy as np
import cv2
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
from collections import deque
import time
import mediapipe as mp

# Load TensorFlow Lite model for emotion recognition
emotion_interpreter = tf.lite.Interpreter(model_path='model_optimized.tflite')
emotion_interpreter.allocate_tensors()
emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Emotion dictionary and colors
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emotion_colors = {
    "Angry": (0, 0, 255),
    "Disgusted": (0, 255, 0),
    "Fearful": (128, 0, 128),
    "Happy": (0, 255, 255),
    "Neutral": (192, 192, 192),
    "Sad": (255, 0, 0),
    "Surprised": (0, 165, 255)
}

class EmotionStabilizer:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.emotion_queue = deque(maxlen=window_size)

    def stabilize(self, emotions):
        self.emotion_queue.append(emotions)
        return {emotion: np.mean([e[emotion] for e in self.emotion_queue]) for emotion in emotion_dict.values()}

class RadiusStabilizer:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.radius_queue = deque(maxlen=window_size)

    def stabilize(self, radius):
        self.radius_queue.append(radius)
        return int(np.mean(self.radius_queue))

class DominantEmotionStabilizer:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.emotion_queue = deque(maxlen=window_size)

    def stabilize(self, emotion):
        self.emotion_queue.append(emotion)
        return max(set(self.emotion_queue), key=self.emotion_queue.count)

class GlobalEmotionStabilizer:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.emotion_queue = deque(maxlen=window_size)

    def stabilize(self, emotions):
        self.emotion_queue.append(emotions)
        stabilized_emotions = {}
        for emotion in emotions.keys():
            values = [e[emotion] for e in self.emotion_queue if emotion in e]
            stabilized_emotions[emotion] = np.mean(values) if values else 0
        return stabilized_emotions

class EmotionalDecorum:
    def __init__(self, initial_value=5, min_value=0, max_value=10):
        self.value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.weights = {
            "Happy": 3,
            "Surprised": 1,
            "Angry": -3,
            "Sad": -2,
            "Disgusted": -2,
            "Fearful": -2,
            "Neutral": 0
        }

    def update(self, emotions):
        decorum_score = sum(self.weights[emotion] * (probability / 100) for emotion, probability in emotions.items())
        normalized_score = (decorum_score + 3) / 6
        self.value = self.min_value + normalized_score * (self.max_value - self.min_value)
        return self.value

@tf.function
def process_face(face):
    face = tf.image.resize(face, (emotion_input_details[0]['shape'][1], emotion_input_details[0]['shape'][2]))
    face = tf.expand_dims(face, axis=0) / 255.0
    return face

def generate_simplified_emotion_aura(frame, center, dominant_emotion, radius, time_factor):
    height, width = frame.shape[:2]
    y, x = np.ogrid[:height, :width]
    
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    pulse_factor = 0.05 * np.sin(time_factor * np.pi) + 1
    normalized_distances = distances / (radius * pulse_factor)
    
    emotional_color = np.array(emotion_colors[dominant_emotion]) / 255.0
    
    aura = np.exp(-normalized_distances**2 / 0.5)[..., np.newaxis] * emotional_color
    
    pulse = np.sin(time_factor * 2) * 0.1 + 0.9
    aura *= pulse
    
    aura = np.clip(aura, 0, 1)
    aura = (aura * 255).astype(np.uint8)
    
    aura = cv2.GaussianBlur(aura, (15, 15), 0)
    
    return cv2.addWeighted(frame, 0.7, aura, 0.8, 0)

def calculate_global_average_emotions(all_emotions):
    if not all_emotions:
        return {}
    average_emotions = {}
    for emotion in emotion_dict.values():
        average_emotions[emotion] = np.mean([emotions[emotion] for emotions in all_emotions if emotion in emotions])
    return average_emotions

def display_global_emotion_info(frame, global_emotions):
    font = cv2.FONT_HERSHEY_SIMPLEX
    sorted_emotions = sorted(global_emotions.items(), key=lambda x: x[1], reverse=True)[:3]

    padding = 5
    box_width = 180
    line_height = 20
    box_height = (len(sorted_emotions) + 1) * line_height + 2 * padding
    start_x, start_y = 10, 10

    cv2.rectangle(frame, (start_x, start_y), (start_x + box_width, start_y + box_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (start_x, start_y), (start_x + box_width, start_y + box_height), (255, 255, 255), 1)

    cv2.putText(frame, "Global emotions", (start_x + padding, start_y + line_height), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for i, (emotion, percentage) in enumerate(sorted_emotions):
        cv2.putText(frame, f"{emotion}: {percentage:.1f}%", (start_x + padding, start_y + (i + 2) * line_height), font, 0.4, emotion_colors[emotion], 1, cv2.LINE_AA)

    return frame

def detect_faces(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            faces.append((x, y, w, h))
    return faces

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
    rounded_score = round(score)  # Arrondir le score à l'entier le plus proche
    cv2.putText(frame, f"{rounded_score}", (badge_x + 5, badge_y + badge_height - 8), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "E.D.", (badge_x + badge_width - 25, badge_y + 12), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return frame

def process_faces(frame, faces, emotion_stabilizers, radius_stabilizers, dominant_emotion_stabilizers, time_factor, emotional_decorum):
    all_emotions = []
    for i, (x, y, w, h) in enumerate(faces):
        # Vérifier si le visage a une taille valide
        if w <= 0 or h <= 0:
            continue  # Passer au visage suivant si la taille n'est pas valide
        
        face = frame[y:y+h, x:x+w]
        
        # Vérifier si l'image du visage est vide
        if face.size == 0:
            continue  # Passer au visage suivant si l'image est vide
        
        try:
            processed_face = process_face(face)
        except tf.errors.InvalidArgumentError:
            print(f"Erreur lors du traitement du visage à la position ({x}, {y}) avec taille ({w}, {h})")
            continue  # Passer au visage suivant en cas d'erreur
        
        emotion_interpreter.set_tensor(emotion_input_details[0]['index'], processed_face.numpy())
        emotion_interpreter.invoke()
        emotion_probs = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])[0]
        
        emotions = {emotion_dict[i]: prob * 100 for i, prob in enumerate(emotion_probs)}
        all_emotions.append(emotions)
        
        if i >= len(emotion_stabilizers):
            emotion_stabilizers.append(EmotionStabilizer())
            radius_stabilizers.append(RadiusStabilizer())
            dominant_emotion_stabilizers.append(DominantEmotionStabilizer())
        
        stabilized_emotions = emotion_stabilizers[i].stabilize(emotions)

        center = (x + w // 2, y + h // 2)
        raw_radius = max(w, h) * 1.1
        stable_radius = radius_stabilizers[i].stabilize(raw_radius)
        
        raw_dominant_emotion = max(emotions, key=emotions.get)
        stable_dominant_emotion = dominant_emotion_stabilizers[i].stabilize(raw_dominant_emotion)
        
        frame = generate_simplified_emotion_aura(frame, center, stable_dominant_emotion, stable_radius, time_factor)
        
        decorum_value = emotional_decorum.update(stabilized_emotions)
        frame = display_compact_stylized_ed_badge(frame, x, y, w, h, decorum_value)

    return frame, all_emotions

def get_projector_resolution():
    # Obtenir la résolution de l'écran principal (qui devrait être le projecteur)
    screen = cv2.getWindowImageRect("Emotion Detection")
    if screen[2] == 0 or screen[3] == 0:  # Si la fenêtre n'existe pas encore
        screen = (0, 0, cv2.getWindowProperty(cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL), cv2.WND_PROP_FULLSCREEN))
    return screen[2], screen[3]

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    # Créer une fenêtre nommée avant d'obtenir sa résolution
    cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)
    
    # Obtenir la résolution du projecteur
    projector_width, projector_height = get_projector_resolution()
    print(f"Projector resolution: {projector_width}x{projector_height}")

    # Mettre la fenêtre en plein écran
    cv2.setWindowProperty("Emotion Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    emotion_stabilizers = []
    radius_stabilizers = []
    dominant_emotion_stabilizers = []
    global_emotion_stabilizer = GlobalEmotionStabilizer()
    emotional_decorum = EmotionalDecorum()

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

        time_factor = (current_time - start_time) % 2

        faces = detect_faces(frame)
        if faces:
            frame, all_emotions = process_faces(frame, faces, emotion_stabilizers, radius_stabilizers, dominant_emotion_stabilizers, time_factor, emotional_decorum)
            
            if all_emotions:
                global_emotions = calculate_global_average_emotions(all_emotions)
                stabilized_global_emotions = global_emotion_stabilizer.stabilize(global_emotions)
                frame = display_global_emotion_info(frame, stabilized_global_emotions)
        else:
            # Afficher un message quand aucun visage n'est détecté
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Redimensionner le frame à la résolution du projecteur
        frame_resized = cv2.resize(frame, (projector_width, projector_height), interpolation=cv2.INTER_AREA)
        
        cv2.imshow('Emotion Detection', frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()