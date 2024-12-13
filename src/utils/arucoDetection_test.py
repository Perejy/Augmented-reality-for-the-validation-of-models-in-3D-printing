import cv2

# Define el diccionario y los parámetros para la detección de ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Inicializa la captura de video con una cámara externa (generalmente índice 1)
cap = cv2.VideoCapture(1)

while True:
    # Captura frame por frame
    ret, frame = cap.read()
    
    if not ret:
        print("No se pudo capturar el frame")
        break

    # Convierte el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta los marcadores en la imagen
    corners, ids, rejectedImgPoints = detector.detectMarkers(frame)

    # Si se detectan marcadores
    if ids is not None:
        for i in range(len(ids)):
            # Dibuja un marco alrededor del marcador detectado
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            # Obtiene las coordenadas del marcador
            c = corners[i][0]
            # Calcula la posición para mostrar el ID (arriba del marcador)
            x = int(c[:, 0].mean())
            y = int(c[:, 1].mean()) - 10
            # Muestra el ID del marcador en la pantalla
            cv2.putText(frame, f'ID: {ids[i][0]}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Muestra el frame resultante
    cv2.imshow('Detección de Marcador ArUco', frame)

    # Rompe el bucle al presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la captura y cierra las ventanas
cap.release()
cv2.destroyAllWindows()