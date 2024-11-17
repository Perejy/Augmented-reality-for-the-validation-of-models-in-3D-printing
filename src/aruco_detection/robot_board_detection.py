import cv2
import numpy as np
import time

def detect_complete_aruco_board(cam_matrix, dist_coeffs, marker_length=0.07, show_rejected=False, refind_strategy=True):
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
        [0, 0, 0], [0.07, 0, 0], [0.07, 0.07, 0], [0, 0.07, 0],  # ID 0
        [18, 0, 0], [18.07, 0, 0], [18.07, 0.07, 0], [18, 0.07, 0],  # ID 1
        [30, -5, 23], [30.07, -5, 23], [30.07, -5 + 0.07, 23], [30, -5 + 0.07, 23],  # ID 2
        [30, -5, 35], [30.07, -5, 35], [30.07, -5 + 0.07, 35], [30, -5 + 0.07, 35],  # ID 3
        [30, 38, 23], [30.07, 38, 23], [30.07, 38 + 0.07, 23], [30, 38 + 0.07, 23],  # ID 4
        [30, 38, 35], [30.07, 38, 35], [30.07, 38 + 0.07, 35], [30, 38 + 0.07, 35],  # ID 5
        [26, -7.5, 45], [26 + 0.07, -7.5, 45], [26 + 0.07, -7.5 + 0.07, 45], [26, -7.5 + 0.07, 45],  # ID 6
        [6, -7, 45], [6 + 0.07, -7, 45], [6 + 0.07, -7 + 0.07, 45], [6, -7 + 0.07, 45],  # ID 7
        [26, 32.5, 45], [26 + 0.07, 32.5, 45], [26 + 0.07, 32.5 + 0.07, 45], [26, 32.5 + 0.07, 45],  # ID 8
        [6, 32.5, 45], [6 + 0.07, 32.5, 45], [6 + 0.07, 32.5 + 0.07, 45], [6, 32.5 + 0.07, 45],  # ID 9
        [-6, -5, 42], [-6 + 0.07, -5, 42], [-6 + 0.07, -5 + 0.07, 42], [-6, -5 + 0.07, 42],  # ID 10
        [-6, -5, 30], [-6 + 0.07, -5, 30], [-6 + 0.07, -5 + 0.07, 30], [-6, -5 + 0.07, 30],  # ID 11
        [-6, 37, 42], [-6 + 0.07, 37, 42], [-6 + 0.07, 37 + 0.07, 42], [-6, 37 + 0.07, 42],  # ID 12
        [-6, 37, 30], [-6 + 0.07, 37, 30], [-6 + 0.07, 37 + 0.07, 30], [-6, 37 + 0.07, 30]  # ID 13
    ], dtype=np.float32)

    # IDs de los marcadores
    ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=np.int32)

    # Crear el objeto Board
    board = cv2.aruco.Board(obj_points.reshape(-1, 4, 3), aruco_dict, ids)

    # Inicializar variables para el tiempo de procesamiento
    total_time = 0
    total_iterations = 0

    while True:
        ret, image = input_video.read()
        if not ret:
            break

        start_time = time.time()

        # Detección de marcadores
        corners, detected_ids, rejected = detector.detectMarkers(image)

        # Convertir rejected a numpy array si es necesario
        if isinstance(rejected, list):
            rejected = np.array(rejected, dtype=object)

        # Estrategia de refinamiento para detectar más marcadores
        if refind_strategy and detected_ids is not None:
            cv2.aruco.refineDetectedMarkers(image, board, corners, detected_ids, rejected, cam_matrix, dist_coeffs)

        # Estimación de la pose del tablero
        if detected_ids is not None and len(detected_ids) > 0:  # Cambiado a >0 para dibujar con al menos un marcador detectado
            img_points = []
            obj_points_detected = []
            for i in range(len(detected_ids)):
                if detected_ids[i] in ids:  # IDs de los marcadores en las esquinas
                    img_points.append(corners[i][0])
                    obj_points_detected.append(obj_points[ids.tolist().index(detected_ids[i]) * 4: (ids.tolist().index(detected_ids[i]) + 1) * 4])
            if len(img_points) >= 1:  # Cambiado a >=1 para dibujar con al menos un marcador detectado
                img_points = np.array(img_points).reshape(-1, 2)
                obj_points_detected = np.array(obj_points_detected).reshape(-1, 3)
                _, rvec, tvec = cv2.solvePnP(obj_points_detected.astype(np.float32), img_points.astype(np.float32), cam_matrix.astype(np.float32), dist_coeffs.astype(np.float32))

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
        if detected_ids is not None:
            cv2.aruco.drawDetectedMarkers(image_copy, corners, detected_ids)
        
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

