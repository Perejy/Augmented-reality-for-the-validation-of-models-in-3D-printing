import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu
import numpy as np
import cv2

def render_object(rvec, tvec, cam_matrix, dist_coeffs, model):
    """
    Renderiza un objeto 3D usando PyOpenGL.
    """
    def draw():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()

        # Configurar la cámara
        view_matrix = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        gl.glMultMatrixf(view_matrix)

        # Aplicar la transformación de la pose estimada
        rot_matrix, _ = cv2.Rodrigues(rvec)
        trans_matrix = np.hstack((rot_matrix, tvec))
        trans_matrix = np.vstack((trans_matrix, [0, 0, 0, 1]))

        gl.glMultMatrixf(trans_matrix.T)

        # Renderizar el modelo
        model.render()

        glut.glutSwapBuffers()

    # Inicializar GLUT
    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_RGBA | glut.GLUT_DOUBLE | glut.GLUT_DEPTH)
    glut.glutInitWindowSize(800, 600)
    glut.glutCreateWindow("3D Object Rendering")

    # Configurar OpenGL
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Configurar la función de dibujo
    glut.glutDisplayFunc(draw)
    glut.glutIdleFunc(draw)

    # Iniciar el bucle principal de GLUT
    glut.glutMainLoop()