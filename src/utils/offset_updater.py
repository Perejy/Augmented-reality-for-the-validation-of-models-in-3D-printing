import json
import tkinter as tk
from tkinter import messagebox

MOVE_STEP = 0.01
ROTATION_STEP = 5  # Rotación en grados

def update_offsets(key, object_offsets, selected_index):
    """
    Función para actualizar los valores de la tupla seleccionada en object_offsets según la tecla presionada.

    Parámetros:
        key: tecla presionada.
        object_offsets: lista de offsets del objeto.
        selected_index: índice del offset actualmente seleccionado.
    """
    offset = object_offsets[selected_index]

    if key == ord('w'):  # Mover arriba
        offset = (offset[0] - MOVE_STEP, offset[1], offset[2])
    elif key == ord('s'):  # Mover abajo
        offset = (offset[0] + MOVE_STEP, offset[1], offset[2])
    elif key == ord('a'):  # Mover izquierda
        offset = (offset[0], offset[1] - MOVE_STEP, offset[2])
    elif key == ord('d'):  # Mover derecha
        offset = (offset[0], offset[1] + MOVE_STEP, offset[2])
    elif key == ord('q'):  # Rotar a la izquierda
        offset = (offset[0], offset[1], offset[2] + ROTATION_STEP)
    elif key == ord('e'):  # Rotar a la derecha
        offset = (offset[0], offset[1], offset[2] - ROTATION_STEP)

    # Actualizar el offset en la lista
    object_offsets[selected_index] = offset

def guardar_offsets(object_offsets):
    """
    Muestra una ventana emergente que pregunta si se desea guardar los offsets de Sancho.
    Si elige sí, sobreescribe el archivo sancho_offsets.json con los nuevos valores.
    Si elige no, los imprime en la consola.
    """
    root = tk.Tk()
    root.withdraw()  # Oculta la ventana principal

    # Preguntar al usuario si quiere guardar los offsets
    respuesta = messagebox.askyesno("Guardar Configuración", 
                                    "¿Quieres guardar la configuración de los offsets para el robot Sancho?")
    
    if respuesta:  # Si el usuario elige sí
        # Crear el diccionario con la estructura requerida
        datos_a_guardar = {"offsets": object_offsets}
        
        # Guardar los offsets en el archivo JSON en la ruta especificada
        ruta = "data/json_data/sancho_offsets.json"
        with open(ruta, "w") as f:
            json.dump(datos_a_guardar, f, indent=4)
        messagebox.showinfo("Guardado", f"Los offsets se han guardado correctamente en {ruta}")
    else:  # Si el usuario elige no
        print("Offsets actuales:")
        print(object_offsets)
    
    root.destroy()  # Cierra la ventana emergente

