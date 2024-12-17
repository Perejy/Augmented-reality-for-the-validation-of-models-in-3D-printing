import cv2
import numpy as np
import json

def load_board_data(json_path, face_number):
    """
    Cargar los puntos del objeto y los IDs para una cara específica desde un archivo JSON.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    face_data = data["faces"][str(face_number)]  # Seleccionar la cara específica
    obj_points = np.array(face_data["obj_points"], dtype=np.float32)
    ids = np.array(face_data["ids"], dtype=np.int32)
    return obj_points, ids