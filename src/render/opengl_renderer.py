import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import cv2
import pygame

def render_object(screen, image_np, rvec, tvec, camMatrix, distCoeffs, model):
    # Configurar la proyección de la cámara
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    
    fx = camMatrix[0, 0]
    fy = camMatrix[1, 1]
    cx = camMatrix[0, 2]
    cy = camMatrix[1, 2]
    near = 0.1
    far = 100.0

    glFrustum((cx - screen.get_width() / 2) / fx * near,
              (cx + screen.get_width() / 2) / fx * near,
              (cy - screen.get_height() / 2) / fy * near,
              (cy + screen.get_height() / 2) / fy * near,
              near, far)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Configurar la vista de la cámara
    rotM = cv2.Rodrigues(rvec)[0]
    tvec = tvec.reshape((3, 1))  # Asegurarse de que tvec tenga la forma correcta
    view_matrix = np.array([[rotM[0][0], rotM[0][1], rotM[0][2], tvec[0][0]],
                            [rotM[1][0], rotM[1][1], rotM[1][2], tvec[1][0]],
                            [rotM[2][0], rotM[2][1], rotM[2][2], tvec[2][0]],
                            [0.0, 0.0, 0.0, 1.0]])

    view_matrix = view_matrix.T
    glLoadMatrixf(view_matrix)

    # Renderizar la imagen de fondo
    glDrawPixels(image_np.shape[1], image_np.shape[0], GL_BGR, GL_UNSIGNED_BYTE, image_np)

    # Renderizar el modelo 3D
    glEnable(GL_DEPTH_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)
    glPushMatrix()
    glTranslatef(0.0, 0.0, -5.0)  # Ajusta la posición del modelo según sea necesario
    glBegin(GL_TRIANGLES)
    for face in model.faces:
        for vertex in face:
            glVertex3fv(model.vertices[vertex - 1])  # Ajuste aquí para evitar el error
    glEnd()
    glPopMatrix()

    glDisable(GL_DEPTH_TEST)

    # Actualizar la pantalla
    pygame.display.flip()