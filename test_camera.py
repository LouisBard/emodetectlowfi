import cv2

def test_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Impossible d'ouvrir la caméra avec l'index {index}")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print(f"Impossible de lire une image de la caméra avec l'index {index}")
        cap.release()
        return False
    
    print(f"Caméra avec l'index {index} fonctionne correctement")
    cap.release()
    return True

# Tester les indices de 0 à 3
for i in range(4):
    test_camera(i)