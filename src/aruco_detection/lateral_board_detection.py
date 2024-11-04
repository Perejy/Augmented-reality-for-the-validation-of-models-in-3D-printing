import cv2
from cv2 import aruco_board as cv2Board
import numpy as np
import time

def detect_lateral_aruco_board(cam_matrix, dist_coeffs, marker_length=0.08, show_rejected=False, refind_strategy=True):
    """
    Función para detectar los marcadores ArUco en tiempo real y estimar su pose.
    """
    # Cargar el diccionario de marcadores y parámetros de detección
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    # Configuración de video
    input_video = cv2.VideoCapture(1)
    wait_time = 10

    # Definir las posiciones de los marcadores en el tablero
    obj_points = np.array([
        [0, 0, 0],  # Esquina superior izquierda
        [0.12, 0, 0],  # Esquina superior derecha (20 cm - 8 cm = 12 cm)
        [0, 0.42, 0],  # Esquina inferior izquierda (50 cm - 8 cm = 42 cm)
        [0.12, 0.42, 0]  # Esquina inferior derecha
    ], dtype=np.float32)

    # IDs de los marcadores
    ids = np.array([2, 3, 4, 5], dtype=np.int32)

    # Crear el objeto Board
    board = cv2Board.aruco.Board(obj_points, aruco_dict, ids)

    # Inicializar variables para el tiempo de procesamiento
    total_time = 0
    total_iterations = 0

    while True:
        ret, image = input_video.read()
        if not ret:
            break

        start_time = time.time()

        # Detección de marcadores
        corners, ids, rejected = detector.detectMarkers(image)

        # Convertir rejected a numpy array si es necesario
        if isinstance(rejected, list):
            rejected = np.array(rejected, dtype=object)

        # Estrategia de refinamiento para detectar más marcadores
        if refind_strategy and ids is not None:
            cv2.aruco.refineDetectedMarkers(image, board, corners, ids, rejected, cam_matrix, dist_coeffs)

        # Estimación de la pose del tablero
        if ids is not None and len(ids) > 0:
            img_points = []
            for i in range(len(ids)):
                if ids[i] in [0, 1, 2, 3]:  # IDs de los marcadores en las esquinas
                    img_points.append(corners[i][0])
            if len(img_points) == 4:
                img_points = np.array(img_points, dtype=np.float32)
                _, rvec, tvec = cv2.solvePnP(obj_points, img_points, cam_matrix, dist_coeffs)

                # Dibujar el eje de referencia
                cv2.drawFrameAxes(image, cam_matrix, dist_coeffs, rvec, tvec, marker_length)

        # Medir el tiempo de procesamiento
        current_time = time.time() - start_time
        total_time += current_time
        total_iterations += 1
        if total_iterations % 30 == 0:
            print(f"Detection Time = {current_time * 1000:.2f} ms (Mean = {1000 * total_time / total_iterations:.2f} ms)")

        # Dibujar resultados
        image_copy = image.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image_copy, corners, ids)
        
        if show_rejected and rejected is not None:
            cv2.aruco.drawDetectedMarkers(image_copy, rejected, borderColor=(100, 0, 255))

        # Mostrar el resultado en pantalla
        cv2.imshow("Board Detection", image_copy)
        key = cv2.waitKey(wait_time) & 0xFF
        if key == 27:  # Presiona 'Esc' para salir
            break

    # Liberar recursos
    input_video.release()
    cv2.destroyAllWindows()

