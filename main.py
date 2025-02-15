import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from src.utils.Calibracion import calibracion
from src.utils.objectpoint_jsonreader import *
from src.aruco_detection.lateral_board_detection import detect_lateral_aruco_board
from src.render.perejy_renderer import *
from src.utils.objCenterer import align_obj
from src.utils.offset_updater import *

# Constante para la ruta del archivo JSON de los puntos de datos
JSON_PATH = "data/json_data/object_points_data.json"

# Configuración de la cámara
calib = calibracion()  
camMatrix, distCoeffs, _ = calib.calibracion_cam()

def seleccionar_configuracion():
    root = tk.Tk()
    root.withdraw()
    respuesta = messagebox.askyesno("Configuración del Robot", "¿Quieres iniciar con la configuración para el robot Sancho?")
    root.destroy()
    return respuesta

def render_if_detected(rvec, tvec, image, obj, cam_matrix, offset, seleccionado):
    if rvec is not None and tvec is not None:
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        projection_matrix = np.hstack((rotation_matrix, tvec))
        projection_matrix = np.dot(cam_matrix, projection_matrix)
        return augment(image, obj, projection_matrix, offset, selected=seleccionado)
    return image

def main():
    usa_sancho = seleccionar_configuracion()

    if usa_sancho:
        object_offsets = load_sancho_offsets()
        if object_offsets is None:
            print("Error cargando los offsets de Sancho. Usando valores por defecto.")
            object_offsets = [(0, 0, 0)] * 4
    else:
        object_offsets = [(0, 0, 0)] * 4

    # Configuración de video
    input_path = "VideoSancho1.mp4"  # Ruta del video de entrada
    output_path = "VideoSancho1_Resultado.mp4"  # Ruta del video de salida

    input_video = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    board_faces = [load_board_data(JSON_PATH, face_num) for face_num in range(1, 5)]

    texture = "data/textures/texture.png"

    align_obj('data/models/front/flower.obj', 'data/models/alineado/banana.obj')
    align_obj('data/models/lateral/lateralSancho.obj', 'data/models/alineado/lateral3.obj')
    align_obj('data/models/back/backSancho.obj', 'data/models/alineado/back.obj')

    objs = [
        three_d_object('data/models/alineado/banana.obj', texture),
        three_d_object('data/models/alineado/lateral3.obj', texture),
        three_d_object('data/models/alineado/back.obj', texture),
        three_d_object('data/models/alineado/lateral3.obj', texture)
    ]

    selected_index = 0

    while True:
        ret, image = input_video.read()
        if not ret:
            break

        for i, ((obj_points, ids), offset) in enumerate(zip(board_faces, object_offsets), start=1):
            rvec, tvec, image = detect_lateral_aruco_board(image, camMatrix, distCoeffs, obj_points, ids, face_id=i)
            seleccionado = (i == selected_index + 1)
            image = render_if_detected(rvec, tvec, image, objs[i-1], camMatrix, offset, seleccionado)

        output_video.write(image)
        cv2.imshow("Board Detection", image)
        key = cv2.waitKey(1) & 0xFF  # Se mantiene un valor bajo para simular video fluido

        if key == 27:  # Esc para salir
            break
        elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
            selected_index = int(chr(key)) - 1
            print(f"Se ha seleccionado la cara {selected_index + 1}")
        elif key in [ord('w'), ord('a'), ord('s'), ord('d'), ord('q'), ord('e')]:
            update_offsets(key, object_offsets, selected_index)
        elif key == ord('p'):
            guardar_offsets(object_offsets)

    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
