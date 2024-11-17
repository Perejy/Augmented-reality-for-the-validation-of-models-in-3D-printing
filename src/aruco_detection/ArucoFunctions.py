import cv2
import cv2.aruco as aruco 
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from utils.Calibracion import *
from ArucoModule import *
from object_module import *
from objloader_simple import *

def main():
    cap = cv2.VideoCapture(1) # Selecciona la entrada de video, 0 para la camara del pc y 1 para cualquier camara externa
    
    calib= calibracion()    # Obejto calibracion 
    CameraMatrix, dist, esquinas = calib.calibracion_cam()    
    
    obj = three_d_object('obj/fox.obj', 'obj/texture.png') # Objeto 3D
    marker = EncuentraAruco("Markers/2.png")  # Devuelve la imagen redimensionada del marcador que he elegido
    while True:
        
        CentersDic = {}
        
        sccuuess, frame = cap.read()  # Toma imagenes de la camara 
        arucoFound = findArucoMarkers(frame, draw=False) # devuelve [bbox,id]
        
        # ## Loop through all the markers and augment each one
        # if len(arucoFound[0])!=0:
        #     for bbox, id in zip(arucoFound[0], arucoFound[1]):        
        #         frame, center = aruco_center(bbox,id,frame, show=True)
        #         CentersDic[id[0]] = center
            
        # frame, centro = arucos_middle(CentersDic,frame, show=True)  
        frame, esquinas_cuadrado = arucos_square(arucoFound, frame, show= True)  

        frame = augmentAruco3D(frame,esquinas_cuadrado, marker, CameraMatrix,obj)      
        
        cv2.imshow("Image", frame)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()