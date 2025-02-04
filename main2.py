import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from src.utils.Calibracion import calibracion
from src.utils.objectpoint_jsonreader import *
from src.aruco_detection.lateral_board_detection import detect_lateral_aruco_board
from src.render.perejy_renderer import *
from src.utils.objCenterer import align_obj
from src.utils.offset_updater import update_offsets

# Constante para la ruta del archivo JSON de los puntos de datos
JSON_PATH = "data/json_data/object_points_data.json"

# Configuración de la cámara
calib = calibracion()  
camMatrix, distCoeffs, _ = calib.calibracion_cam()

def seleccionar_configuracion():
    """
    Función para mostrar una ventana emergente y preguntar por la configuración de Sancho.
    Retorna True si el usuario elige "Sí", False si elige "No".
    """
    root = tk.Tk()
    root.withdraw()  # Oculta la ventana principal

    respuesta = messagebox.askyesno("Configuración del Robot", 
                                    "¿Quieres iniciar con la configuración para el robot Sancho?")
    root.destroy()  # Cierra la ventana emergente
    return respuesta

def render_if_detected(rvec, tvec, image, obj, cam_matrix, offset, seleccionado):
    if rvec is not None and tvec is not None:
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        projection_matrix = np.hstack((rotation_matrix, tvec))
        projection_matrix = np.dot(cam_matrix, projection_matrix)
        return augment(image, obj, projection_matrix, offset, selected=seleccionado)
    return image

def main():
    # Preguntar al usuario si quiere la configuración de Sancho
    usa_sancho = seleccionar_configuracion()

    # Cargar offsets según la elección
    if usa_sancho:
        object_offsets = load_sancho_offsets()
        if object_offsets is None:
            print("Error cargando los offsets de Sancho. Usando valores por defecto.")
            object_offsets = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
    else:
        object_offsets = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]  # Configuración por defecto
    
    # Configuración de video
    input_video = cv2.VideoCapture(1)
    wait_time = 10

    # Cargar los object points e ids de las distintas caras
    board_faces = [load_board_data(JSON_PATH, face_num) for face_num in range(1, 5)]

    # Preparar los modelos
    texture = "data/textures/texture.png"
    
    align_obj('data/models/front/banana.obj', 'data/models/alineado/banana.obj')
    align_obj('data/models/lateral/lateral3.obj', 'data/models/alineado/lateral3.obj')
    align_obj('data/models/back/back.obj', 'data/models/alineado/back.obj')

    obj1 = three_d_object('data/models/alineado/banana.obj', texture)
    obj2 = three_d_object('data/models/alineado/lateral3.obj', texture)
    obj3 = three_d_object('data/models/alineado/back.obj', texture)
    obj4 = three_d_object('data/models/alineado/lateral3.obj', texture)
    objs = [obj1, obj2, obj3, obj4]

    selected_index = 0  # Índice de la cara seleccionada

    while True:
        ret, image = input_video.read()
        if not ret:
            break

        for i, ((obj_points, ids), offset) in enumerate(zip(board_faces, object_offsets), start=1):
            rvec, tvec, image = detect_lateral_aruco_board(image, camMatrix, distCoeffs, obj_points, ids, face_id=i)
            seleccionado = (i == selected_index + 1)
            image = render_if_detected(rvec, tvec, image, objs[i-1], camMatrix, offset, seleccionado)

        cv2.imshow("Board Detection", image)
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == 27:  # Esc para salir
            break
        elif key == ord('1'):
            selected_index = 0
            print("Se ha seleccionado la cara frontal del robot")
        elif key == ord('2'):
            selected_index = 1
            print("Se ha seleccionado la cara izquierda del robot")
        elif key == ord('3'):
            selected_index = 2
            print("Se ha seleccionado la cara trasera del robot")
        elif key == ord('4'):
            selected_index = 3
            print("Se ha seleccionado la cara derecha del robot")
        elif key in [ord('w'), ord('a'), ord('s'), ord('d'), ord('q'), ord('e')]:  # Movimiento WASD y rotacion QE
            update_offsets(key, object_offsets, selected_index)
        elif key == ord('p'):
            print("object_offsets:", object_offsets)
    
    input_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()