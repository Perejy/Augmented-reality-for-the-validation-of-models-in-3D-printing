import cv2
import cv2.aruco as aruco 
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from Calibracion import *
from ArucoModule import *

def main():
    cap = cv2.VideoCapture(1) # Selecciona la entrada de video, 0 para la camara del pc y 1 para cualquier camara externa
    
    calib= calibracion()    # Obejto calibracion 
    CameraMatrix, dist, esquinas = calib.calibracion_cam()    
    
    ##imgAug = cv2.imread("Markers/23.jpg")
    
    augDics = loadAugImages("Markers")
    while True:
        sccuuess, frame = cap.read()  # Toma imagenes de la camara 
        arucoFound = findArucoMarkers(frame, draw=False) # devuelve [bbox,id]
        
        ## Loop through all the markers and augment each one
        if len(arucoFound[0])!=0:
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                if int(id[0]) in augDics.keys():
                    ##frame = augmentAruco(bbox, id, frame, augDics[int(id)])
                    frame = aruco_center(bbox,id,frame)
                    

        
        cv2.imshow("Image", frame)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()