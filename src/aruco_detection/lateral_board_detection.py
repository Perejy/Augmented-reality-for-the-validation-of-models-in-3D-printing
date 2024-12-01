import cv2
import numpy as np

def detect_lateral_aruco_board(image, cam_matrix, dist_coeffs, marker_length=0.07, refind_strategy=False):
    """
    Función para detectar los marcadores ArUco en una imagen y estimar su pose.
    """
    # Cargar el diccionario de marcadores y parámetros de detección
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    # Definir las posiciones de los marcadores en el tablero
    obj_points = np.array([
        [0, 0, 0], [0.07, 0, 0], [0.07, 0.07, 0], [0, 0.07, 0],  # Esquina superior izquierda
        [0.12, 0, 0], [0.19, 0, 0], [0.19, 0.07, 0], [0.12, 0.07, 0],  # Esquina superior derecha
        [0, 0.42, 0], [0.07, 0.42, 0], [0.07, 0.49, 0], [0, 0.49, 0],  # Esquina inferior izquierda
        [0.12, 0.42, 0], [0.19, 0.42, 0], [0.19, 0.49, 0], [0.12, 0.49, 0]  # Esquina inferior derecha
    ], dtype=np.float32)

    # IDs de los marcadores
    ids = np.array([2, 3, 4, 5], dtype=np.int32)

    # Crear el objeto Board
    board = cv2.aruco.Board(obj_points.reshape(-1, 4, 3), aruco_dict, ids)

    # Detección de marcadores
    corners, detected_ids, rejected = detector.detectMarkers(image)

    # Convertir rejected a numpy array si es necesario
    if isinstance(rejected, list):
        rejected = np.array(rejected, dtype=object)

    # Estrategia de refinamiento para detectar más marcadores
    if refind_strategy and detected_ids is not None:
        cv2.aruco.ArucoDetector.refineDetectedMarkers(image, board, corners, detected_ids, rejected, cam_matrix, dist_coeffs)

    # Estimación de la pose del tablero
    if detected_ids is not None and len(detected_ids) > 0:
        img_points = []
        obj_points_detected = []
        for i in range(len(detected_ids)):
            if detected_ids[i] in ids:
                img_points.append(corners[i][0])
                obj_points_detected.append(obj_points[ids.tolist().index(detected_ids[i]) * 4: (ids.tolist().index(detected_ids[i]) + 1) * 4])
        if len(img_points) >= 1:
            img_points = np.array(img_points).reshape(-1, 2)
            obj_points_detected = np.array(obj_points_detected).reshape(-1, 3)
            _, rvec, tvec = cv2.solvePnP(obj_points_detected.astype(np.float32), img_points.astype(np.float32), cam_matrix.astype(np.float32), dist_coeffs.astype(np.float32))
            cv2.drawFrameAxes(image, cam_matrix, dist_coeffs, rvec, tvec, marker_length)
            return rvec, tvec, image

    return None, None, image