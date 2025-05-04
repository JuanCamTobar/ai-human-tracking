import cv2
import mediapipe as mp
import csv
import numpy as np

# Ruta del video y archivo de salida
video_path = "C:\\Users\\juan_\\OneDrive - Universidad Icesi\\Escritorio\\Proyectos IA\\videos\\video1.mp4"
csv_path = "datos_promediados22.csv"

# Configuración de MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Agrupaciones de puntos por región
REGIONES = {
    "cara": list(range(0, 10)),
    "hombros_cadera": [11, 12, 23, 24],
    "brazo_izquierdo": [11, 13, 15, 17, 19, 21],
    "brazo_derecho": [12, 14, 16, 18, 20, 22],
    "pierna_izquierda": [23, 25, 27, 29, 31],
    "pierna_derecha": [24, 26, 28, 30, 32]
}

# Función para calcular promedio de una región
def get_avg(landmarks, indices):
    x = np.mean([landmarks[i].x for i in indices])
    y = np.mean([landmarks[i].y for i in indices])
    z = np.mean([landmarks[i].z for i in indices])
    return x, y, z

# Inicializar video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# CSV
f = open(csv_path, 'w', newline='')
writer = csv.writer(f)
header = ['segundo']
for region in REGIONES:
    header.extend([f'{region}_x', f'{region}_y', f'{region}_z'])
writer.writerow(header)

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    seconds = frame_num / fps

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark

        # Dibujar conexiones
        #mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        row = [round(seconds, 2)]

        for region, indices in REGIONES.items():
            x, y, z = get_avg(landmarks, indices)
            row.extend([x, y, z])
            # Dibujo en la imagen
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

        writer.writerow(row)

    cv2.imshow('Pose promediada', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Liberar
cap.release()
f.close()
cv2.destroyAllWindows()
