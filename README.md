# Augmented-reality-for-the-validation-of-models-in-3D-printing

Este Trabajo de Fin de Grado propone una solución basada en realidad aumentada para la validación previa de modelos 3D antes de su impresión, aplicándolo al robot móvil Sancho como caso de estudio. 

La aplicación desarrollada permite la superposición en tiempo real de los modelos sobre la plataforma robótica, garantizando su correcta escala y ajuste mediante técnicas de visión por computador y el uso de marcadores ArUco. 


## Librerias a instalar 

Recomiendo instalar un entorno virtual para la instalación de librerias, en mi caso hice uso de venv.
Los comandos para la instalación del entorno virtual y las librerias son los siguientes:
```
python3 -m venv arucolib
arucolib\Scripts\activate  
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

### Nota
El proyecto está configurado para usarlo con una cámara externa, para ello usé la aplicación *droidcam* disponible en android con la que enlacé la cámara del movil al ordenador. Por otro lado, tambien recomendaría modificar las imagenes de calibrado por unas tomadas por la cámara que se vaya a usar.

Otra opción es usar una cámara interna del propio ordenador como la de los portátiles, para ello modificar la siguiente linea de código:
```
cap = cv2.VideoCapture(1) # Selecciona la entrada de video, 0 para la camara del pc y 1 para cualquier camara externa.
```
