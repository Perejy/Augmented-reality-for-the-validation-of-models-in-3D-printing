import cv2
import numpy as np
from src.utils.Calibracion import calibracion
from src.utils.objectpoint_jsonreader import load_board_data
from src.aruco_detection.lateral_board_detection import detect_lateral_aruco_board
from src.render.perejy_renderer import *

# Constante para la ruta del archivo JSON de los puntos de datos
JSON_PATH = "data/json_data/object_points_data.json"

# Configuración de la cámara
calib = calibracion()  # Objeto calibracion 
camMatrix, distCoeffs, _ = calib.calibracion_cam()

def render_if_detected(rvec, tvec, image, obj, cam_matrix, offset):
    """
    Función auxiliar para convertir rvec/tvec a proyección y renderizar el objeto.
    """
    if rvec is not None and tvec is not None:
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        projection_matrix = np.hstack((rotation_matrix, tvec))
        projection_matrix = np.dot(cam_matrix, projection_matrix)
        return augment(image, obj, projection_matrix, offset)
    return image

def main():
    # Configuración de video
    input_video = cv2.VideoCapture(1)
    wait_time = 10

    # Cargar los object points e ids de las distintas caras en una lista
    board_faces = [
        load_board_data(JSON_PATH, face_num) for face_num in range(1, 5)
    ]
    
    # Posiciones del objeto 3D para cada cara (offsets) [cara 1 (h,w), cara 2, cara 3, cara 4]
    object_offsets = [(0, 0.1), (0.2, 0.05), (0.25, 0.12), (0.2, 0.05)]

    # Cargar el modelo 3D
    obj = three_d_object('data/models/charmander.obj', "data/textures/texture.png")

    while True:
        ret, image = input_video.read()
        if not ret:
            break

        # Procesar cada cara del tablero
        for i, ((obj_points, ids), offset) in enumerate(zip(board_faces, object_offsets), start=1):
            rvec, tvec, image = detect_lateral_aruco_board(image, camMatrix, distCoeffs, obj_points, ids)
            image = render_if_detected(rvec, tvec, image, obj, camMatrix, offset)

        # Mostrar el resultado en pantalla
        cv2.imshow("Board Detection", image)
        key = cv2.waitKey(wait_time) & 0xFF
        if key == 27:  # Presiona 'Esc' para salir
            break

    # Liberar recursos
    input_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
