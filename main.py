import cv2
import numpy as np
from src.aruco_detection.board_detection import detect_aruco_board
from src.aruco_detection.lateral_board_detection import detect_lateral_aruco_board
from src.aruco_detection.robot_board_detection import detect_complete_aruco_board
from src.utils.Calibracion import calibracion
from src.render.model_loader import load_model
from src.render.opengl_renderer import render_object

# Configuración de la cámara
calib = calibracion()  # Objeto calibracion 
camMatrix, distCoeffs, _ = calib.calibracion_cam()

def main():
    # Configuración de video
    input_video = cv2.VideoCapture(1)
    wait_time = 10

    # Cargar el modelo 3D
    model = load_model("data/models/fox.obj")

    while True:
        ret, image = input_video.read()
        if not ret:
            break

        # Detectar marcadores ArUco y obtener la pose
        rvec, tvec = detect_lateral_aruco_board(image, camMatrix, distCoeffs)

        if rvec is not None and tvec is not None:
            # Renderizar el objeto 3D
            render_object(rvec, tvec, camMatrix, distCoeffs, model)

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