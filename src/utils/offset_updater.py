MOVE_STEP = 0.01

def update_offsets(key, object_offsets, selected_index):
    """
    Función para actualizar los valores de la tupla seleccionada en object_offsets según la tecla presionada.

    Parámetros:
        key: tecla presionada.
        object_offsets: lista de offsets del objeto.
        selected_index: índice del offset actualmente seleccionado.
    """
    offset = object_offsets[selected_index]

    if key == ord('w'):  # Flecha arriba
        offset = (offset[0] - MOVE_STEP, offset[1])
    elif key == ord('s'):  # Flecha abajo
        offset = (offset[0] + MOVE_STEP, offset[1])
    elif key == ord('a'):  # Flecha izquierda
        offset = (offset[0], offset[1] - MOVE_STEP)
    elif key == ord('d'):  # Flecha derecha
        offset = (offset[0], offset[1] + MOVE_STEP)

    # Actualizar el offset en la lista
    object_offsets[selected_index] = offset