import cv2
import numpy as np

# Diccionarios globales para el suavizado (un historial para cada cara)
prev_rvec_dict = {}
prev_tvec_dict = {}
alpha = 0.8  # Peso para el filtro exponencial

def detect_lateral_aruco_board(image, cam_matrix, dist_coeffs, obj_points, ids, marker_length=0.07, refind_strategy=False, face_id=None):
    """
    Función para detectar los marcadores ArUco en una imagen y estimar su pose.
    """
    global prev_rvec_dict, prev_tvec_dict, alpha

    # Inicializar los valores previos para esta cara si no existen
    if face_id not in prev_rvec_dict:
        prev_rvec_dict[face_id] = None
        prev_tvec_dict[face_id] = None

    # Cargar el diccionario de marcadores y parámetros de detección
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

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
            
            # Recalcular el eje Z
            rmat, _ = cv2.Rodrigues(rvec)
            x_axis = rmat[:, 0]  # Eje X
            y_axis = rmat[:, 1]  # Eje Y
            
            # Calcular el eje Z como el producto cruzado entre X e Y
            z_axis = np.cross(x_axis, y_axis)
            z_axis /= np.linalg.norm(z_axis)
            
            # Reconstruir la matriz de rotación
            rmat[:, 2] = z_axis
            rvec, _ = cv2.Rodrigues(rmat)

            # Suavizar rvec y tvec
            if prev_rvec_dict[face_id] is not None and prev_tvec_dict[face_id] is not None:
                rvec = alpha * prev_rvec_dict[face_id] + (1 - alpha) * rvec
                tvec = alpha * prev_tvec_dict[face_id] + (1 - alpha) * tvec

            # Guardar los valores actuales para esta cara
            prev_rvec_dict[face_id] = rvec
            prev_tvec_dict[face_id] = tvec

            # Dibujar los ejes ajustados
            cv2.drawFrameAxes(image, cam_matrix, dist_coeffs, rvec, tvec, marker_length)
            return rvec, tvec, image

    return None, None, image