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
