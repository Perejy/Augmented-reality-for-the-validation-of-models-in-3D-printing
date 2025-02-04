import cv2
import numpy as np
import json

SANCHO_OFFSETS_PATH = "data/json_data/sancho_offsets.json" 

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

def load_sancho_offsets():
    """
    Carga los offsets del archivo JSON.
    Si el archivo no existe o hay un error, devuelve None.
    """
    try:
        with open(SANCHO_OFFSETS_PATH, 'r') as f:
            data = json.load(f)
            return data.get("offsets", None)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error cargando offsets de Sancho: {e}")
        return None