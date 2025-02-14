def align_obj(file_path, output_path):
    vertices = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Leer vértices
    for line in lines:
        if line.startswith('v '):  # Solo líneas de vértices
            parts = line.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    # Calcular el mínimo Z (suelo), el mínimo X (izquierda) y el mínimo Y (parte inferior)
    min_z = min(v[2] for v in vertices)
    min_x = min(v[0] for v in vertices)
    min_y = min(v[1] for v in vertices)

    # Crear nuevo archivo .obj con vértices alineados
    with open(output_path, 'w') as f:
        for line in lines:
            if line.startswith('v '):
                parts = line.split()
                new_vertex = [
                    float(parts[1]) - min_x,  # Llevar X a la izquierda
                    float(parts[2]) - min_y,  # Llevar Y a la parte inferior
                    float(parts[3]) - min_z,  # Alinear con el suelo
                ]
                f.write(f"v {new_vertex[0]} {new_vertex[1]} {new_vertex[2]}\n")
            else:
                f.write(line)

