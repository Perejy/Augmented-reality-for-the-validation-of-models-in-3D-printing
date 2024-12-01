import OpenGL.GL as gl

class Model:
    def __init__(self):
        self.vertices = []
        self.faces = []

    def load_model(self):
        """
        Carga un modelo 3D desde un archivo .obj.
        """
        file_path = "data/models/fox.obj"
        with open(file_path) as file:
            for line in file:
                if line.startswith('v '):
                    parts = line.split()
                    vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                    self.vertices.append(vertex)
                elif line.startswith('f '):
                    parts = line.split()
                    face = []
                    for part in parts[1:]:
                        vertex_index = part.split('/')[0]
                        face.append(int(vertex_index) - 1)
                    self.faces.append(face)
        
        print(f"Loaded {len(self.vertices)} vertices and {len(self.faces)} faces.")

    def render(self):
        gl.glBegin(gl.GL_TRIANGLES)
        for face in self.faces:
            for vertex in face:
                gl.glVertex3fv(self.vertices[vertex])
        gl.glEnd()

def load_model(file_path):
    model = Model()
    model.load_model()
    return model