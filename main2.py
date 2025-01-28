import cv2
import numpy as np
from src.utils.Calibracion import calibracion
from src.utils.objectpoint_jsonreader import load_board_data
from src.aruco_detection.lateral_board_detection import detect_lateral_aruco_board
from src.render.perejy_renderer import *
from src.utils.objCenterer import align_obj_to_ground
from src.utils.offset_updater import update_offsets

# Constante para la ruta del archivo JSON de los puntos de datos
JSON_PATH = "data/json_data/object_points_data.json"

# Configuración de la cámara
calib = calibracion()  # Objeto calibracion 
camMatrix, distCoeffs, _ = calib.calibracion_cam()

def render_if_detected(rvec, tvec, image, obj, cam_matrix, offset, seleccionado):
    """
    Función auxiliar para convertir rvec/tvec a proyección y renderizar el objeto.
    """
    if rvec is not None and tvec is not None:
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        projection_matrix = np.hstack((rotation_matrix, tvec))
        projection_matrix = np.dot(cam_matrix, projection_matrix)
        return augment(image, obj, projection_matrix, offset, selected=seleccionado)
    return image

def main():
    # Configuración de video
    input_video = cv2.VideoCapture(1)
    wait_time = 10

    # Cargar los object points e ids de las distintas caras en una lista
    board_faces = [
        load_board_data(JSON_PATH, face_num) for face_num in range(1, 5)
    ]
    
    # Posiciones del objeto 3D para cada cara (offsets) [cara 1 (h,w), cara 2, cara 3, cara 4]
    #object_offsets = [(0, 0.1), (0.2, 0.05), (0.25, 0.12), (0.2, 0.05)]
    object_offsets = [(0, 0), (0, 0), (0, 0), (0, 0)]
    
    texture = "data/textures/texture.png"
    
    align_obj_to_ground('data/models/front/banana.obj', 'data/models/alineado/banana.obj')
    align_obj_to_ground('data/models/lateral/lateral3.obj', 'data/models/alineado/lateral3.obj')
    align_obj_to_ground('data/models/back/back.obj', 'data/models/alineado/back.obj')
    
    # Cargar el modelo 3D
    obj1 = three_d_object('data/models/alineado/banana.obj', texture)
    obj2 = three_d_object('data/models/alineado/lateral3.obj', texture)
    obj3 = three_d_object('data/models/alineado/back.obj', texture)
    obj4 = three_d_object('data/models/alineado/lateral3.obj', texture)
    objs = [obj1,obj2,obj3,obj4]
    
    # Variable para rastrear la tupla seleccionada
    selected_index = 0  # Índice por defecto

    while True:
        ret, image = input_video.read()
        if not ret:
            break

        # Procesar cada cara del tablero
        for i, ((obj_points, ids), offset) in enumerate(zip(board_faces, object_offsets), start=1):
            rvec, tvec, image = detect_lateral_aruco_board(image, camMatrix, distCoeffs, obj_points, ids, face_id=i)
            seleccionado = False
            if i == selected_index + 1 :
                seleccionado = True
            image = render_if_detected(rvec, tvec, image, objs[i-1], camMatrix, offset, seleccionado)

        # Mostrar el resultado en pantalla
        cv2.imshow("Board Detection", image)
        key = cv2.waitKey(wait_time) & 0xFF
        
        # Manejo de eventos del teclado
        if key == 27:  # Presiona 'Esc' para salir
            break
        elif key == ord('1'):  # Tecla 1
            selected_index = 0
            print("Se ha seleccionado la cara frontal del robot")
        elif key == ord('2'):  # Tecla 2
            selected_index = 1
            print("Se ha seleccionado la cara izquierda del robot")
        elif key == ord('3'):  # Tecla 3
            selected_index = 2
            print("Se ha seleccionado la cara trasera del robot")
        elif key == ord('4'):  # Tecla 4
            selected_index = 3
            print("Se ha seleccionado la cara derecha del robot")
        elif key in [ord('w'), ord('a'), ord('s'), ord('d')]:  # Teclas WASD
            update_offsets(key, object_offsets, selected_index)
        elif key == ord('p'):  # Imprimir los valores actuales
            print("object_offsets:", object_offsets)
    

    # Liberar recursos
    input_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
