# main.py

import numpy as np
#from aruco_detection.board_detection import detect_aruco_board
from src.aruco_detection.lateral_board_detection import detect_lateral_aruco_board
from src.utils.Calibracion import calibracion

# Configuración de la cámara
calib= calibracion()    # Obejto calibracion 
camMatrix,distCoeffs,_ = calib.calibracion_cam()

def main():
    # Llama a la función de detección de tablero ArUco
    ## detect_aruco_board(camMatrix,distCoeffs)
    detect_lateral_aruco_board(camMatrix, distCoeffs)

if __name__ == "__main__":
    main()