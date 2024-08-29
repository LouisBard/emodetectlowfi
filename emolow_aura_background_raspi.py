import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter
from collections import deque
import time
import mediapipe as mp

# Load TensorFlow Lite model for emotion recognition
emotion_interpreter = Interpreter(model_path='model_optimized.tflite')
emotion_interpreter.allocate_tensors()
emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3, model_selection=1)

# Emotion dictionary and colors
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emotion_colors = {
    "Angry": (0, 0, 255), "Disgusted": (0, 255, 0), "Fearful": (128, 0, 128),
    "Happy": (0, 255, 255), "Neutral": (192, 192, 192), "Sad": (255, 0, 0),
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
            return avg_value if np.any(np.abs(new_value - avg_value) > self.threshold) else new_value
        return avg_value if abs(new_value - avg_value) > self.threshold else new_value

class Person:
    def __init__(self):
        self.filters = {
            'emotion': {emotion: SmoothingFilter(window_size=10, threshold=3) for emotion in emotion_dict.values()},
            'decorum': SmoothingFilter(window_size=20, threshold=0.3),
            'aura_color': SmoothingFilter(window_size=10, threshold=10),
        }
        self.last_seen = time.time()

    def update(self, emotions, x, y, w, h, current_time):
        self.last_seen = current_time
        smoothed_emotions = {emotion: self.filters['emotion'][emotion].update(value) for emotion, value in emotions.items()}
        dominant_emotion = max(smoothed_emotions, key=smoothed_emotions.get)
        smooth_decorum = self.filters['decorum'].update(calculate_decorum(smoothed_emotions))
        smooth_color = self.filters['aura_color'].update(np.array(emotion_colors[dominant_emotion]))
        return smoothed_emotions, dominant_emotion, smooth_decorum, smooth_color, (x, y, w, h)

class GlobalEmotionStabilizer:
    def __init__(self, window_size=30):
        self.emotion_queue = deque(maxlen=window_size)

    def update(self, emotions):
        self.emotion_queue.append(emotions)
        return {emotion: np.mean([e[emotion] for e in self.emotion_queue if emotion in e]) for emotion in emotion_dict.values()}

def calculate_decorum(emotions):
    weights = {"Happy": 3, "Surprised": 1, "Angry": -3, "Sad": -2, "Disgusted": -2, "Fearful": -2, "Neutral": 0}
    return (sum(weights[emotion] * (probability / 100) for emotion, probability in emotions.items()) + 3) / 6 * 10

def process_face(face):
    # Ensure the face image is in RGB order (OpenCV uses BGR by default)
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    # Resize the face image to match the input shape of the model
    resized_face = cv2.resize(face_rgb, (emotion_input_details[0]['shape'][1], emotion_input_details[0]['shape'][2]), interpolation=cv2.INTER_LINEAR)
    
    # Convert to float32 and normalize to [0, 1]
    normalized_face = resized_face.astype(np.float32) / 255.0
    
    # Simple contrast enhancement (adjusted to match tf.image.adjust_contrast)
    mean = np.mean(normalized_face, axis=(0, 1), keepdims=True)
    enhanced_face = (normalized_face - mean) * 1.5 + mean
    enhanced_face = np.clip(enhanced_face, 0, 1)
    
    # Apply average pooling
    kernel = np.ones((3,3), np.float32) / 9
    blurred = cv2.filter2D(enhanced_face, -1, kernel)
    
    # Simple sharpening using a basic unsharp mask
    sharpened_face = np.clip(enhanced_face * 2 - blurred, 0, 1)
    
    # Add batch dimension for the final output
    return np.expand_dims(sharpened_face, 0).astype(np.float32)

def generate_simplified_emotion_aura(frame, center, color, radius, time_factor):
    height, width = frame.shape[:2]
    y, x = np.ogrid[:height, :width]
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    pulse_factor = 0.05 * np.sin(time_factor * np.pi) + 1
    normalized_distances = distances / (radius * pulse_factor)
    aura = np.exp(-normalized_distances**2 / 0.5)[..., np.newaxis] * (np.array(color) / 255.0)
    aura *= np.sin(time_factor * 2) * 0.1 + 0.9
    return cv2.addWeighted(frame, 0.7, cv2.GaussianBlur((np.clip(aura, 0, 1) * 255).astype(np.uint8), (15, 15), 0), 0.8, 0)

def display_global_emotion_info(frame, global_emotions):
    font = cv2.FONT_HERSHEY_SIMPLEX
    sorted_emotions = sorted(global_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
    padding, box_width, line_height = 5, 180, 20
    box_height = (len(sorted_emotions) + 1) * line_height + 2 * padding
    start_x, start_y = 10, 10
    cv2.rectangle(frame, (start_x, start_y), (start_x + box_width, start_y + box_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (start_x, start_y), (start_x + box_width, start_y + box_height), (255, 255, 255), 1)
    cv2.putText(frame, "Global emotions", (start_x + padding, start_y + line_height), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for i, (emotion, percentage) in enumerate(sorted_emotions):
        cv2.putText(frame, f"{emotion}: {percentage:.1f}%", (start_x + padding, start_y + (i + 2) * line_height), font, 0.4, emotion_colors[emotion], 1, cv2.LINE_AA)
    return frame

def detect_faces(frame):
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.detections:
        ih, iw = frame.shape[:2]
        return [(int(detection.location_data.relative_bounding_box.xmin * iw),
                 int(detection.location_data.relative_bounding_box.ymin * ih),
                 int(detection.location_data.relative_bounding_box.width * iw),
                 int(detection.location_data.relative_bounding_box.height * ih))
                for detection in results.detections]
    return []

def display_compact_dystopian_ed_badge(frame, x, y, w, h, score):
    badge_width, badge_height = 100, 40
    badge_x = min(x + w + 5, frame.shape[1] - badge_width - 5)
    badge_y = min(y, frame.shape[0] - badge_height - 5)
    
    if score >= 7:
        color = (0, 255, 0)  # Vert vif
        status = "COMPLIANT"
    elif score >= 4:
        color = (0, 255, 255)  # Jaune vif
        status = "ACCEPTABLE"
    else:
        color = (0, 0, 255)  # Rouge vif
        status = "VIOLATION"
    
    glitch_offset = int(np.sin(time.time() * 10) * 2)
    
    cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_width, badge_y + badge_height), (50, 50, 50), -1)
    cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_width, badge_y + badge_height), color, 1)
    
    cv2.rectangle(frame, (badge_x + glitch_offset, badge_y), (badge_x + badge_width + glitch_offset, badge_y + badge_height), color, 1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "E.D.", (badge_x + 5, badge_y + 15), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    
    cv2.putText(frame, f"{score:.1f}", (badge_x + 5, badge_y + 32), font, 0.5, color, 1, cv2.LINE_AA)
    
    status_font_size = 0.3
    status_width = cv2.getTextSize(status, font, status_font_size, 1)[0][0]
    status_x = badge_x + badge_width - status_width - 3
    cv2.putText(frame, status, (status_x, badge_y + 32), font, status_font_size, color, 1, cv2.LINE_AA)
    
    return frame

def process_faces(frame, faces, people, current_time):
    all_emotions = []
    active_ids = set()
    for face in faces:
        x, y, w, h = face
        if w <= 0 or h <= 0 or y < 0 or x < 0 or y+h > frame.shape[0] or x+w > frame.shape[1]:
            continue
        try:
            face_img = frame[max(0, y):min(y+h, frame.shape[0]), max(0, x):min(x+w, frame.shape[1])]
            if face_img.size == 0:
                continue
            processed_face = process_face(face_img)
            emotion_interpreter.set_tensor(emotion_input_details[0]['index'], processed_face)
            emotion_interpreter.invoke()
            emotions = {emotion_dict[i]: prob * 100 for i, prob in enumerate(emotion_interpreter.get_tensor(emotion_output_details[0]['index'])[0])}
            all_emotions.append(emotions)
        except Exception as e:
            print(f"Error processing face at ({x}, {y}): {str(e)}")
            continue
        person_id = find_closest_person(people, x, y)
        if person_id not in people:
            people[person_id] = Person()
        person = people[person_id]
        smoothed_emotions, dominant_emotion, smooth_decorum, smooth_color, _ = person.update(emotions, x, y, w, h, current_time)
        frame = generate_simplified_emotion_aura(frame, (x + w // 2, y + h // 2), smooth_color, max(w, h) * 1.1, current_time % 2)
        frame = display_compact_dystopian_ed_badge(frame, x, y, w, h, smooth_decorum)
        active_ids.add(person_id)
    people = {pid: person for pid, person in people.items() if current_time - person.last_seen < 1.0}
    return frame, all_emotions, people

def find_closest_person(people, x, y, max_distance=50):
    closest_id = min(people.keys(), key=lambda pid: ((int(pid.split('_')[0]) - x) ** 2 + (int(pid.split('_')[1]) - y) ** 2) ** 0.5, default=None)
    return closest_id if closest_id and ((int(closest_id.split('_')[0]) - x) ** 2 + (int(closest_id.split('_')[1]) - y) ** 2) ** 0.5 <= max_distance else f"{x}_{y}"

def load_background_image(image_path):
    background = cv2.imread(image_path)
    if background is None:
        raise ValueError(f"Impossible de charger l'image de fond: {image_path}")
    return cv2.resize(background, (1920, 1080))

def overlay_frame_on_background(background, frame):
    # Redimensionner le cadre d'analyse
    overlay_height, overlay_width = int(background.shape[0] * 0.7), int(background.shape[1] * 0.7)
    frame_resized = cv2.resize(frame, (overlay_width, overlay_height))
    
    # Positionner le cadre
    y_offset = (background.shape[0] - overlay_height) // 2
    x_offset = (background.shape[1] - overlay_width) // 2
    
    # Créer une vue de l'arrière-plan au lieu d'une copie
    result = background
    
    # Superposer le cadre redimensionné
    result[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = frame_resized
    
    # Texte dystopique
    text = "IS YOUR EMOTIONAL DECORUM SCORE SUFFICIENT TO AVOID SOCIAL EXILE?"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 2
    text_color = (200, 200, 200)
    
    # Obtenir la taille du texte et le positionner
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_width, text_height = text_size
    
    # Diviser le texte en deux lignes si nécessaire
    if text_width > result.shape[1] - 40:
        words = text.split()
        mid = len(words) // 2
        text1 = " ".join(words[:mid])
        text2 = " ".join(words[mid:])
        
        text_size1, _ = cv2.getTextSize(text1, font, font_scale, font_thickness)
        text_size2, _ = cv2.getTextSize(text2, font, font_scale, font_thickness)
        
        text_x1 = (result.shape[1] - text_size1[0]) // 2
        text_x2 = (result.shape[1] - text_size2[0]) // 2
        text_y1 = y_offset + overlay_height + 40
        text_y2 = text_y1 + text_height + 10
        
        # Ajouter un fond noir semi-transparent
        cv2.rectangle(result, (0, text_y1 - 30), (result.shape[1], text_y2 + 10), (0, 0, 0), -1)
        
        # Ajouter le texte principal
        cv2.putText(result, text1, (text_x1, text_y1), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(result, text2, (text_x2, text_y2), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    else:
        text_x = (result.shape[1] - text_width) // 2
        text_y = y_offset + overlay_height + 40
        
        # Ajouter un fond noir semi-transparent
        cv2.rectangle(result, (0, text_y - 30), (result.shape[1], text_y + 10), (0, 0, 0), -1)
        
        # Ajouter le texte principal
        cv2.putText(result, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    # Ajouter un léger bruit pour un effet plus brut (optimisé)
    noise = np.random.randint(0, 10, result.shape[:2], dtype=np.uint8)
    result = cv2.add(result, cv2.merge([noise, noise, noise]))
    
    return result

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la webcam.")
        return
    
    cv2.namedWindow("Analyse Émotionnelle Dystopique", cv2.WINDOW_NORMAL)
    screen = cv2.getWindowImageRect("Analyse Émotionnelle Dystopique")
    projector_width, projector_height = screen[2:] if screen[2] and screen[3] else (1920, 1080)  # Résolution par défaut si non détectée
    cv2.setWindowProperty("Analyse Émotionnelle Dystopique", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Charger l'image de fond
    background = load_background_image("fond_distopic.jpg")
    
    people = {}
    global_emotion_stabilizer = GlobalEmotionStabilizer(window_size=30)
    start_time = time.time()
    frame_count = 0
    last_fps_time = start_time
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur: Impossible de lire l'image de la webcam.")
            break
        
        frame_count += 1
        current_time = time.time()
        if current_time - last_fps_time >= 1:
            print(f"FPS: {frame_count / (current_time - last_fps_time):.2f}")
            frame_count, last_fps_time = 0, current_time
        
        faces = detect_faces(frame)
        if faces:
            frame, all_emotions, people = process_faces(frame, faces, people, current_time)
            if all_emotions:
                global_emotions = {emotion: np.mean([e[emotion] for e in all_emotions if emotion in e]) for emotion in emotion_dict.values()}
                stabilized_global_emotions = global_emotion_stabilizer.update(global_emotions)
                frame = display_global_emotion_info(frame, stabilized_global_emotions)
        else:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Superposer le cadre d'analyse sur l'arrière-plan
        result_frame = overlay_frame_on_background(background, frame)
        
        cv2.imshow('Analyse Émotionnelle Dystopique', result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()