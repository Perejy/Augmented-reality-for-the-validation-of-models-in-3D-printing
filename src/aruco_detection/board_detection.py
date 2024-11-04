import cv2
import numpy as np
import time

def detect_aruco_board(cam_matrix, dist_coeffs, markers_x=3, markers_y=2, marker_length=0.06, marker_separation=0.03, show_rejected=False, refind_strategy=True):
    """
    Función para detectar el tablero de marcadores ArUco en tiempo real y estimar su pose.
    """
    # Cargar el diccionario de marcadores y parámetros de detección
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    # Configuración de video
    input_video = cv2.VideoCapture(1)
    wait_time = 10

    # Longitud del eje de referencia
    axis_length = 0.5 * (min(markers_x, markers_y) * (marker_length + marker_separation) + marker_separation)

    # Crear objeto GridBoard
    board = cv2.aruco.GridBoard((markers_x, markers_y), marker_length, marker_separation, aruco_dict)

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

        # Estrategia de refinamiento para detectar más marcadores
        if refind_strategy:
            cv2.aruco.refineDetectedMarkers(image, board, corners, ids, rejected, cam_matrix, dist_coeffs)

        # Estimación de la pose del tablero
        markers_of_board_detected = 0
        if ids is not None and len(ids) > 0:
            obj_points, img_points = board.matchImagePoints(corners, ids)
            if obj_points is not None and img_points is not None:
                _, rvec, tvec = cv2.solvePnP(obj_points, img_points, cam_matrix, dist_coeffs)
                markers_of_board_detected = int(obj_points.shape[0] / 4)

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
        
        if show_rejected and rejected:
            cv2.aruco.drawDetectedMarkers(image_copy, rejected, borderColor=(100, 0, 255))

        # Dibujar el eje de referencia si se ha detectado el tablero
        if markers_of_board_detected > 0:
            cv2.drawFrameAxes(image_copy, cam_matrix, dist_coeffs, rvec, tvec, axis_length)

        # Mostrar el resultado en pantalla
        cv2.imshow("Board Detection", image_copy)
        key = cv2.waitKey(wait_time) & 0xFF
        if key == 27:  # Presiona 'Esc' para salir
            break

    # Liberar recursos
    input_video.release()
    cv2.destroyAllWindows()
