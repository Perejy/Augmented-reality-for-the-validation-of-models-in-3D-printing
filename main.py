import cv2
import numpy as np
import pygame
import pygame.camera
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from src.aruco_detection.board_detection import detect_aruco_board
from src.aruco_detection.lateral_board_detection import detect_lateral_aruco_board
from src.aruco_detection.robot_board_detection import detect_complete_aruco_board
from src.utils.Calibracion import calibracion
from src.render.model_loader import load_model
from src.render.opengl_renderer import render_object

# Configuración de la cámara
calib = calibracion()  # Objeto calibracion 
camMatrix, distCoeffs, _ = calib.calibracion_cam()

def main():
    # Inicializar Pygame
    pygame.init()
    pygame.camera.init()

    # Listar todas las cámaras disponibles
    cam_list = pygame.camera.list_cameras()
    if not cam_list:
        print("¡No se encontró ninguna cámara!")
        exit()

    cam_index = 1  # Cambia este índice según la cámara que desees usar

    # Configurar y empezar la cámara seleccionada
    cam = pygame.camera.Camera(cam_list[cam_index], (640, 480))
    cam.start()

    # Configurar la pantalla
    screen = pygame.display.set_mode((640, 480), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Camera Stream")

    # Cargar el modelo 3D
    model = load_model("data/models/fox.obj")

    # Bucle principal
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
        # Capturar imagen de la cámara
        image_surface = cam.get_image()
        image_string = pygame.image.tostring(image_surface, 'RGB')
        image_np = np.frombuffer(image_string, dtype=np.uint8).reshape((480, 640, 3))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Detectar marcadores ArUco y obtener la pose
        rvec, tvec, image_np = detect_lateral_aruco_board(image_np, camMatrix, distCoeffs)

        if rvec is not None and tvec is not None:
             # Renderizar el objeto 3D y mostrar la detección de ArUco
             render_object(screen, image_np, rvec, tvec, camMatrix, distCoeffs, model)

        # Convertir numpy.ndarray a pygame.Surface
        image_surface = pygame.surfarray.make_surface(image_np)

        # Renderizar la imagen en OpenGL
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawPixels(640, 480, GL_BGR, GL_UNSIGNED_BYTE, image_np)

        # Mostrar la imagen en la pantalla
        pygame.display.flip()

    cam.stop()
    pygame.quit()

if __name__ == "__main__":
    main()