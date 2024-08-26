import numpy as np
import cv2
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
from collections import deque
import time
import mediapipe as mp
import logging

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Chargement du modèle TensorFlow Lite pour la reconnaissance des émotions
emotion_interpreter = tf.lite.Interpreter(model_path='model_optimized.tflite')
emotion_interpreter.allocate_tensors()
emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()

# Initialisation de MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3, model_selection=1)

# Dictionnaire des émotions et couleurs
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

class SmoothingFilter:
    def __init__(self, window_size=10, threshold=0.1):
        self.window_size = window_size
        self.threshold = threshold
        self.values = deque(maxlen=window_size)

    def update(self, new_value):
        self.values.append(new_value)
        avg_value = np.mean(self.values, axis=0)
        if isinstance(new_value, np.ndarray):
            if np.any(np.abs(new_value - avg_value) > self.threshold):
                return avg_value
        else:
            if abs(new_value - avg_value) > self.threshold:
                return avg_value
        return new_value

class Person:
    def __init__(self):
        self.emotion_filters = {emotion: SmoothingFilter(window_size=10, threshold=3) for emotion in emotion_dict.values()}
        self.decorum_filter = SmoothingFilter(window_size=20, threshold=0.3)
        self.aura_color_filter = SmoothingFilter(window_size=10, threshold=10)
        self.position_filter = SmoothingFilter(window_size=3, threshold=10)
        self.size_filter = SmoothingFilter(window_size=3, threshold=5)
        self.last_seen = time.time()

    def update(self, emotions, x, y, w, h, current_time):
        self.last_seen = current_time
        
        smoothed_emotions = {emotion: self.emotion_filters[emotion].update(value) 
                             for emotion, value in emotions.items()}
        
        dominant_emotion = max(smoothed_emotions, key=smoothed_emotions.get)
        
        decorum_value = calculate_decorum(smoothed_emotions)
        smooth_decorum = self.decorum_filter.update(decorum_value)
        
        raw_color = np.array(emotion_colors[dominant_emotion])
        smooth_color = self.aura_color_filter.update(raw_color)
        
        smooth_x = self.position_filter.update(x)
        smooth_y = self.position_filter.update(y)
        smooth_w = self.size_filter.update(w)
        smooth_h = self.size_filter.update(h)
        
        return (smoothed_emotions, dominant_emotion, smooth_decorum, 
                smooth_color, (smooth_x, smooth_y, smooth_w, smooth_h))

class GlobalEmotionStabilizer:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.emotion_queue = deque(maxlen=window_size)

    def update(self, emotions):
        self.emotion_queue.append(emotions)
        stabilized_emotions = {}
        for emotion in emotion_dict.values():
            values = [e[emotion] for e in self.emotion_queue if emotion in e]
            stabilized_emotions[emotion] = np.mean(values) if values else 0
        return stabilized_emotions

def calculate_decorum(emotions):
    weights = {
        "Happy": 3, "Surprised": 1, "Angry": -3, "Sad": -2,
        "Disgusted": -2, "Fearful": -2, "Neutral": 0
    }
    decorum_score = sum(weights[emotion] * (probability / 100) for emotion, probability in emotions.items())
    return (decorum_score + 3) / 6 * 10

@tf.function
def process_face(face):
    face = tf.image.resize(face, (emotion_input_details[0]['shape'][1], emotion_input_details[0]['shape'][2]))
    face = tf.expand_dims(face, axis=0) / 255.0
    return face

def generate_simplified_emotion_aura(frame, center, color, radius, time_factor):
    height, width = frame.shape[:2]
    y, x = np.ogrid[:height, :width]
    
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    pulse_factor = 0.05 * np.sin(time_factor * np.pi) + 1
    normalized_distances = distances / (radius * pulse_factor)
    
    emotional_color = np.array(color) / 255.0
    
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
    
    global_emotions = {}
    for emotion in emotion_dict.values():
        values = [emotions[emotion] for emotions in all_emotions if emotion in emotions]
        global_emotions[emotion] = np.mean(values) if values else 0
    
    return global_emotions

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


def find_closest_person(people, x, y, max_distance=50):
    closest_id = None
    min_dist = float('inf')
    for pid, person in people.items():
        px, py, _, _ = person.position_filter.values[-1] if person.position_filter.values else (0, 0, 0, 0)
        dist = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
        if dist < min_dist and dist <= max_distance:
            closest_id = pid
            min_dist = dist
    return closest_id if closest_id is not None else f"{x}_{y}"

def display_compact_stylized_ed_badge(frame, x, y, w, h, score):
    badge_width, badge_height = 60, 30
    badge_x, badge_y = int(x + w + 5), int(y)  # Assurez-vous que les coordonnées sont des entiers

    frame_height, frame_width = frame.shape[:2]
    if badge_x + badge_width > frame_width:
        badge_x = int(x - badge_width - 5)
    if badge_y + badge_height > frame_height:
        badge_y = int(y + h - badge_height)

    base_color = (0, 200, 0) if score >= 7 else (0, 200, 200) if score >= 4 else (0, 0, 200)

    overlay = frame.copy()
    cv2.rectangle(overlay, (badge_x, badge_y), (badge_x + badge_width, badge_y + badge_height), base_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_width, badge_y + badge_height), base_color, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    rounded_score = round(score)
    cv2.putText(frame, f"{rounded_score}", (badge_x + 5, badge_y + badge_height - 8), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "E.D.", (badge_x + badge_width - 25, badge_y + 12), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return frame

def process_faces(frame, faces, people, current_time):
    all_emotions = []
    active_ids = set()
    
    for face in faces:
        x, y, w, h = face
        
        if w <= 0 or h <= 0 or frame[y:y+h, x:x+w].size == 0:
            continue

        face_img = frame[y:y+h, x:x+w]
        
        try:
            processed_face = process_face(face_img)
            emotion_interpreter.set_tensor(emotion_input_details[0]['index'], processed_face.numpy())
            emotion_interpreter.invoke()
            emotion_probs = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])[0]
            emotions = {emotion_dict[i]: prob * 100 for i, prob in enumerate(emotion_probs)}
            
            logger.debug(f"Processing face at ({x}, {y}), size: {w}x{h}")
            logger.debug(f"Detected emotions: {emotions}")
            
            person_id = find_closest_person(people, x, y)
            if person_id not in people:
                people[person_id] = Person()
            
            person = people[person_id]
            update_result = person.update(emotions, x, y, w, h, current_time)
            
            if isinstance(update_result, tuple) and len(update_result) == 5:
                smoothed_emotions, dominant_emotion, smooth_decorum, smooth_color, smooth_position = update_result
            else:
                logger.error(f"Unexpected update result for person {person_id}: {update_result}")
                continue
            
            logger.debug(f"Person ID: {person_id}, Dominant emotion: {dominant_emotion}")
            
            # Utilisez smooth_position pour l'affichage
            sx, sy, sw, sh = smooth_position
            
            # Générer l'aura et afficher le badge pour ce visage spécifique
            center = (int(sx + sw // 2), int(sy + sh // 2))
            frame = generate_simplified_emotion_aura(frame, center, smooth_color, max(sw, sh) * 1.1, current_time % 2)
            frame = display_compact_stylized_ed_badge(frame, sx, sy, sw, sh, smooth_decorum)
            
            all_emotions.append(emotions)
            active_ids.add(person_id)
        except Exception as e:
            logger.error(f"Error processing face at ({x}, {y}): {str(e)}")
            continue

    # Supprimer les personnes inactives
    people = {pid: person for pid, person in people.items() if current_time - person.last_seen < 1.0}

    return frame, all_emotions, people

def get_projector_resolution():
    screen = cv2.getWindowImageRect("Emotion Detection")
    if screen[2] == 0 or screen[3] == 0:
        screen = (0, 0, cv2.getWindowProperty(cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL), cv2.WND_PROP_FULLSCREEN))
    return screen[2], screen[3]

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Unable to open webcam.")
        return

    cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)
    projector_width, projector_height = get_projector_resolution()
    logger.info(f"Projector resolution: {projector_width}x{projector_height}")
    cv2.setWindowProperty("Emotion Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    people = {}
    global_emotion_stabilizer = GlobalEmotionStabilizer(window_size=30)

    start_time = time.time()
    frame_count = 0
    last_fps_time = start_time

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Unable to read webcam frame.")
            break

        frame_count += 1
        current_time = time.time()
        if current_time - last_fps_time >= 1:
            fps = frame_count / (current_time - last_fps_time)
            logger.info(f"FPS: {fps:.2f}")
            frame_count = 0
            last_fps_time = current_time

        faces = detect_faces(frame)
        if faces:
            frame, all_emotions, people = process_faces(frame, faces, people, current_time)
            
            if all_emotions:
                global_emotions = calculate_global_average_emotions(all_emotions)
                stabilized_global_emotions = global_emotion_stabilizer.update(global_emotions)
                frame = display_global_emotion_info(frame, stabilized_global_emotions)
        else:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame_resized = cv2.resize(frame, (projector_width, projector_height), interpolation=cv2.INTER_AREA)
        
        cv2.imshow('Emotion Detection', frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()