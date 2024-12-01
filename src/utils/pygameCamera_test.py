import pygame
import pygame.camera

# Inicializar Pygame
pygame.init()
pygame.camera.init()

# Listar todas las cámaras disponibles
cam_list = pygame.camera.list_cameras()
if not cam_list:
    print("¡No se encontró ninguna cámara!")
    exit()

# Mostrar las cámaras disponibles y seleccionar una
print("Cámaras disponibles:")
for i, cam in enumerate(cam_list):
    print(f"{i}: {cam}")

# Seleccionar la cámara externa (por ejemplo, la segunda cámara en la lista)
cam_index = 1  # Cambia este índice según la cámara que desees usar
if cam_index >= len(cam_list):
    print("Índice de cámara no válido")
    exit()

# Configurar y empezar la cámara seleccionada
cam = pygame.camera.Camera(cam_list[cam_index], (640, 480))
cam.start()

# Configurar la pantalla
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Camera Stream")

# Bucle principal
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Capturar imagen de la cámara
    image = cam.get_image()

    # Mostrar la imagen en la pantalla
    screen.blit(image, (0, 0))
    pygame.display.update()

# Detener la cámara y salir de Pygame
cam.stop()
pygame.quit()