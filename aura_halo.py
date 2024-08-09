import numpy as np
import cv2
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
from collections import deque
import multiprocessing
import time

# Load TensorFlow Lite model for emotion recognition
emotion_interpreter = tf.lite.Interpreter(model_path='model_optimized.tflite')
emotion_interpreter.allocate_tensors()
emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Emotion color dictionary with more vivid colors
emotion_colors = {
    "Angry": (0, 0, 255),      # Pure Red
    "Disgusted": (0, 255, 0),  # Pure Green
    "Fearful": (255, 0, 0),    # Pure Blue
    "Happy": (0, 255, 255),    # Pure Yellow
    "Neutral": (255, 128, 0),  # Bright Orange
    "Sad": (255, 0, 255),      # Pure Magenta
    "Surprised": (255, 255, 0) # Pure Cyan
}

class EmotionStabilizer:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.emotion_queue = deque(maxlen=window_size)

    def stabilize(self, emotions):
        self.emotion_queue.append(emotions)
        avg_emotions = {}
        for emotion in emotion_dict.values():
            avg_emotions[emotion] = np.mean([e[emotion] for e in self.emotion_queue])
        return avg_emotions

class ColorStabilizer:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.color_queue = deque(maxlen=window_size)

    def stabilize(self, colors):
        self.color_queue.append(colors)
        avg_colors = np.mean(self.color_queue, axis=0).astype(int)
        return [tuple(color) for color in avg_colors]

class RadiusStabilizer:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.radius_queue = deque(maxlen=window_size)

    def stabilize(self, radius):
        self.radius_queue.append(radius)
        return int(np.mean(self.radius_queue))

def optimize_face_detection(gray, scaleFactor=1.3, minNeighbors=5):
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    return facecasc.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

def generate_oval_pulsating_halo(frame, center, colors, radius, time_factor):
    height, width = frame.shape[:2]
    halo = np.zeros((height, width, 3), dtype=np.float32)
    
    y, x = np.ogrid[:height, :width]
    
    aspect_ratio = 1.5
    distances = np.sqrt((x - center[0])**2 + ((y - center[1])/aspect_ratio)**2)
    
    pulse_factor = 0.05 * np.sin(time_factor * np.pi) + 1
    normalized_distances = distances / (radius * pulse_factor)
    
    num_colors = len(colors)
    for i in range(num_colors):
        inner_radius = i / num_colors
        outer_radius = (i + 1) / num_colors
        color = colors[i]
        
        if i == 0:  # For the innermost layer (primary emotion)
            layer_mask = normalized_distances < outer_radius
        else:
            layer_mask = np.logical_and(normalized_distances >= inner_radius, normalized_distances < outer_radius)
        
        layer_gradient = np.clip((outer_radius - normalized_distances) / (outer_radius - inner_radius), 0, 1)
        layer_gradient = layer_gradient ** 1.5
        
        saturated_color = np.clip(np.array(color) * 1.5, 0, 255)  # Increased color saturation
        halo += np.outer(layer_mask * layer_gradient, saturated_color).reshape(height, width, 3)
    
    halo = np.clip(halo / np.max(halo), 0, 1)
    halo = cv2.GaussianBlur(halo, (31, 31), 0)
    
    result = cv2.addWeighted(frame, 1, (halo * 255).astype(np.uint8), 0.8, 0)  # Increased opacity
    
    return result

def optimize_display_emotion_info(frame, emotions):
    font = cv2.FONT_HERSHEY_SIMPLEX
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]

    for i, (emotion, percentage) in enumerate(sorted_emotions):
        color = emotion_colors[emotion]
        text = f"{emotion}: {percentage:.2f}%"
        cv2.putText(frame, text, (10, 30 + i * 30), font, 0.6, color, 2, cv2.LINE_AA)

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
    color_stabilizer = ColorStabilizer()
    radius_stabilizer = RadiusStabilizer()
    input_shape = emotion_input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read webcam frame.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        faces = optimize_face_detection(gray)

        face_data = [(small_frame[y:y+h, x:x+w], input_width, input_height) for (x, y, w, h) in faces]
        emotions_list = pool.starmap(process_face, face_data)

        if emotions_list:
            stabilized_emotions = emotion_stabilizer.stabilize(emotions_list[0])
            frame = optimize_display_emotion_info(frame, stabilized_emotions)

        current_time = time.time()
        time_factor = (current_time - start_time) % 2

        for (x, y, w, h), emotions in zip(faces, emotions_list):
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            if len(sorted_emotions) >= 3:
                colors = [emotion_colors[emotion] for emotion, _ in sorted_emotions]
                
                stabilized_colors = color_stabilizer.stabilize(colors)
                
                center = ((x + w // 2) * 2, (y + h // 2) * 2)
                raw_radius = max(w, h) * 2.2
                stable_radius = radius_stabilizer.stabilize(raw_radius)
                
                frame = generate_oval_pulsating_halo(frame, center, stabilized_colors, stable_radius, time_factor)

        cv2.imshow('Emotion Detection', cv2.resize(frame, (1024, 768), interpolation=cv2.INTER_LANCZOS4))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()