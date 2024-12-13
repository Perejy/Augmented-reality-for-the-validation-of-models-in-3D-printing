import cv2
import numpy as np
from src.utils.Calibracion import calibracion
from src.aruco_detection.lateral_board_detection1 import detect_lateral_aruco_board1
from src.aruco_detection.lateral_board_detection2 import detect_lateral_aruco_board2
from src.aruco_detection.lateral_board_detection3 import detect_lateral_aruco_board3
from src.aruco_detection.lateral_board_detection4 import detect_lateral_aruco_board4
from src.render.perejy_renderer import *

# Configuración de la cámara
calib = calibracion()  # Objeto calibracion 
camMatrix, distCoeffs, _ = calib.calibracion_cam()

def main():
    # Configuración de video
    input_video = cv2.VideoCapture(1)
    wait_time = 10

    # Cargar el modelo 3D
    # obj = three_d_object('data/models/fox.obj', "data/textures/texture.png")
    obj = three_d_object('data/models/charmander.obj', "data/textures/texture.png")

    while True:
        ret, image = input_video.read()
        if not ret:
            break

        # Detectar marcadores ArUco y obtener la pose
        rvec1, tvec1, image = detect_lateral_aruco_board1(image, camMatrix, distCoeffs)
        rvec2, tvec2, image = detect_lateral_aruco_board2(image, camMatrix, distCoeffs)
        rvec3, tvec3, image = detect_lateral_aruco_board3(image, camMatrix, distCoeffs)
        rvec4, tvec4, image = detect_lateral_aruco_board4(image, camMatrix, distCoeffs)

        if rvec1 is not None and tvec1 is not None:
            # Renderizar el objeto 3D
            ##render_object(rvec, tvec, camMatrix, distCoeffs, model)
            
            # Convertir rvec a matriz de rotación
            rotation_matrix, _ = cv2.Rodrigues(rvec1)

            # Crear la matriz de proyección (3x4) combinando la matriz de rotación y el vector de traslación
            projection_matrix = np.hstack((rotation_matrix, tvec1))
            projection_matrix = np.dot(camMatrix,projection_matrix)
            # Imprimir la matriz de proyección
            #print("Matriz de Proyección:",projection_matrix)
             
            image = augment(image,obj,projection_matrix,(0,0.1))
        
        if rvec2 is not None and tvec2 is not None:
            # Renderizar el objeto 3D
            ##render_object(rvec, tvec, camMatrix, distCoeffs, model)
            
            # Convertir rvec a matriz de rotación
            rotation_matrix, _ = cv2.Rodrigues(rvec2)

            # Crear la matriz de proyección (3x4) combinando la matriz de rotación y el vector de traslación
            projection_matrix = np.hstack((rotation_matrix, tvec2))
            projection_matrix = np.dot(camMatrix,projection_matrix)
            # Imprimir la matriz de proyección
            #print("Matriz de Proyección:",projection_matrix)
             
            image = augment(image,obj,projection_matrix,(0.2,0.05))
            
        if rvec3 is not None and tvec3 is not None:
            # Renderizar el objeto 3D
            ##render_object(rvec, tvec, camMatrix, distCoeffs, model)
            
            # Convertir rvec a matriz de rotación
            rotation_matrix, _ = cv2.Rodrigues(rvec3)

            # Crear la matriz de proyección (3x4) combinando la matriz de rotación y el vector de traslación
            projection_matrix = np.hstack((rotation_matrix, tvec3))
            projection_matrix = np.dot(camMatrix,projection_matrix)
            # Imprimir la matriz de proyección
            #print("Matriz de Proyección:",projection_matrix)
             
            image = augment(image,obj,projection_matrix,(0.25,0.12))
            
        if rvec4 is not None and tvec4 is not None:
            # Renderizar el objeto 3D
            ##render_object(rvec, tvec, camMatrix, distCoeffs, model)
            
            # Convertir rvec a matriz de rotación
            rotation_matrix, _ = cv2.Rodrigues(rvec4)

            # Crear la matriz de proyección (3x4) combinando la matriz de rotación y el vector de traslación
            projection_matrix = np.hstack((rotation_matrix, tvec4))
            projection_matrix = np.dot(camMatrix,projection_matrix)
            # Imprimir la matriz de proyección
            #print("Matriz de Proyección:",projection_matrix)
             
            image = augment(image,obj,projection_matrix,(0.2,0.05))

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