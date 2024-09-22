import cv2
import cv2.aruco as aruco 
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from Calibracion import *

def findArucoMarkers(img, markerSize = 6, totalMarkers = 250, draw = True, drawGray = False):
    """Devuelve un array con las posiciones de las esquinas y el id del aruco encontrado"""
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgGray = binarize_kmeans(gaussian_smoothing(imgGray,0.2,1),5)
    if drawGray:
        cv2.imshow("gray", imgGray)
    key = getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.getPredefinedDictionary(key)
    arucoParam = aruco.DetectorParameters()
    bboxs, ids, rejected = cv2.aruco.detectMarkers(imgGray,arucoDict,parameters = arucoParam) 
    
    
    if draw:
        aruco.drawDetectedMarkers(img,bboxs)
        
        
    return [bboxs,ids]

def findArucoMarkers3d(img,matrix,dist, markerSize = 6, totalMarkers = 250, draw = True):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgGray = binarize_kmeans(gaussian_smoothing(imgGray,0.2,1),5)
    cv2.imshow("gray", imgGray)
    key = getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = cv2.aruco.detectMarkers(imgGray,arucoDict,parameters = arucoParam) # ,cameraMatrix = matrix, distCoeff = dist
    
    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(bboxs, 0.02, matrix, dist)
    ##(rvec-tvec).any()
    if draw:
        aruco.drawDetectedMarkers(img,bboxs)
        print(matrix)
        cv2.drawFrameAxes(img,matrix, dist, rvec,tvec,0.01)
        
        
    return [bboxs,ids]

def loadAugImages(path):
    myList = os.listdir(path)
    noOfMarkers = len(myList)
    print("Total Number of Markers Detected: ", noOfMarkers)
    augDics = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f'{path}/{imgPath}')
        augDics[key] = imgAug
    return augDics

def augmentAruco(bbox, id, img, imgAug, drawID = True):
    
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]
    
    h, w, c = imgAug.shape
    
    pts1 = np.array([tl,tr,br,bl])
    pts2 = np.float32([[0,0],[w,0],[w,h],[0,h]])
    matrix, _ = cv2.findHomography(pts2,pts1)
    imgOut = cv2.warpPerspective(imgAug,matrix,(img.shape[1],img.shape[0]))
    cv2.fillConvexPoly(img,pts1.astype(int), (0,0,0)) 
    imgOut = img + imgOut
    
    if drawID:
        cv2.putText(imgOut, str(id), (int(tl[0]),int(tl[1])), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
    
    return imgOut


def arucos_square(arucos, frame, show = False):
    ### arucos es un array con todos las esquinas y otro array con todos los ids, concuerdan entre ellos
    """ Ejemplo con dos arucos
    [(array([[[331., 398.],
        [ 74., 408.],
        [100., 173.],
        [321., 170.]]], dtype=float32), 
        array([[[630., 376.],
        [404., 387.],
        [384., 179.],
        [582., 172.]]], dtype=float32)), 
        array([[ 2],
       [23]], dtype=int32)]
    {2: (206, 287), 23: (500, 278)}
    
    
     Args: 
        arucos: array con todos las esquinas y otro array con todos los ids
        frame: imagen del frame
        show(false): Muestra el cuadrado

    Returns:
        frame: Imagen resultante  
        max_array: Las esquinas del cuadrado generado [tl_max,tr_max,br_max,bl_max]
        
    """
    CentersDic = {}
    tl_list = []
    tr_list = []
    br_list = []
    bl_list = []
    
    tl_max = (0,0)
    tr_max = (0,0)
    br_max = (0,0)
    bl_max = (0,0)
    
    if len(arucos[0])!=0:
        for bbox, id in zip(arucos[0], arucos[1]):        
            frame, center = aruco_center(bbox,id,frame, show=True)
            CentersDic[id[0]] = center
            tl_list.append((int(bbox[0][0][0]), int(bbox[0][0][1])))
            tr_list.append((int(bbox[0][1][0]), int(bbox[0][1][1])))
            br_list.append((int(bbox[0][2][0]), int(bbox[0][2][1])))
            bl_list.append((int(bbox[0][3][0]), int(bbox[0][3][1])))
        ## Revisar esto de igualar todo a center
        tl_max = center
        tr_max = center
        br_max = center
        bl_max = center        
    
    _, centroide = arucos_middle(CentersDic, frame, show=True)
    
    corners_array = [tl_list,tr_list,br_list,bl_list]
    max_array = [tl_max,tr_max,br_max,bl_max]
    i = 0
    while i< 4: ## Cuatro porque lo recorro una vez por cada esquina [tl,tr,br,bl]
        xx_list = corners_array[i]
        
        j = 0
        if len(xx_list) == 1:
            max_array[i] = xx_list[0]   
        elif len(xx_list) > 1:
            max = manhattan(xx_list[j], centroide)
            max_array[i] = xx_list[j]
            j +=1
            while j < len(xx_list):
                distance = manhattan(xx_list[j], centroide)
                if max < distance:
                    max = distance
                    max_array[i] = xx_list[j]
                j +=1      
        i += 1;       
    
    if show:
        v1, v2 = max_array[0][0],max_array[0][1]
        v3, v4 = max_array[1][0],max_array[1][1]
        v5, v6 = max_array[2][0],max_array[2][1]
        v7, v8 = max_array[3][0],max_array[3][1]
        
        # Cara Inferior
        cv2.line(frame, (int(v1), int(v2)), (int(v3), int(v4)), (255,255,0),3)
        cv2.line(frame, (int(v5), int(v6)), (int(v7), int(v8)), (255,255,0),3)
        cv2.line(frame, (int(v1), int(v2)), (int(v7), int(v8)), (255,255,0),3)
        cv2.line(frame, (int(v3), int(v4)), (int(v5), int(v6)), (255,255,0),3)
    
    return frame, max_array


def arucos_middle(CenterDic, img, show = False):
    """ Calcula el centroide de todos los centros de los ArUcos
    
    Args: 
        CenterDic: Diccionario de los centros de los aruco {id: (x,y)}
        img: imagen del frame
        show(false): Muestra el centroide

    Returns:
        imgOut: Imagen resultante  
        center: devuelve la posición del centroide de los ArUcos (x,y)
        
    """
    center = None
    num_centers = len(CenterDic)
    ListCenter = list(CenterDic.values()) # Recoge la lista de centros del diccionario
    
    if num_centers == 1:
        center = ListCenter[0]
    elif num_centers > 1:
        sumax = 0
        sumay = 0
        
        for center in ListCenter:
            sumax += center[0]
            sumay += center[1]
        
        center = (int(sumax/num_centers), int(sumay/num_centers))

    if show:
        imgOut = cv2.circle(img, center, 4, color=(255, 0, 255), thickness = -1)
    else:
        imgOut = img

    return imgOut, center

def aruco_center(bbox, id, img, show = False):
    """ Calcula la posición de los centros de los ArUcos.
    
    Args: 
        bbox: Esquinas de los Arucos sin formato
        id: identificador del ArUco
        img: imagen del frame
        show(false): Muestra los puntos de las esquinas en la imagen

    Returns:
        imgOut: Imagen resultante  
        center: devuelve la posición del centro del ArUco (x,y)
        
    """
    
    imgOut, corners = aruco_corners(bbox,id, img, show)
    
    center = middle_corners_point(corners)
    
    if show:
        imgOut = cv2.circle(imgOut, center, 4, color=(0, 0, 255), thickness = -1)
    
    return imgOut, center

def aruco_corners(bbox, id, img, show=False):
    """ Calcula la posición de las esquinas de los ArUcos. Si show es true se muestran en pantalla con puntos rojos
    
    Args: 
        bbox: Esquinas de los Arucos sin formato
        id: identificador del ArUco
        img: imagen del frame
        show(false): Muestra los puntos de las esquinas en la imagen

        
    Returns:
        imgOut: Imagen resultante
        corners: array con las cuatro esquinas de un aruco [tl,tr,bl,br] en int    
    """
    
    
    tl = (int(bbox[0][0][0]), int(bbox[0][0][1]))
    tr = (int(bbox[0][1][0]), int(bbox[0][1][1]))
    br = (int(bbox[0][2][0]), int(bbox[0][2][1]))
    bl = (int(bbox[0][3][0]), int(bbox[0][3][1]))
    
    corners = [tl,tr,bl,br]
    imgOut = img

    if show:
        for corner in corners:
            imgOut = cv2.circle(imgOut, corner, 4, color=(0, 255, 0), thickness = -1)
    
    return imgOut, corners

def middle_corners_point(corners):
    """ Calcula el punto medio de 4 puntos
    
    Args: 
        corners: array con las cuatro esquinas de un aruco [tl,tr,bl,br] en int

        
    Returns:
        MiddlePoint: punto medio de los 4 puntos
        
        Calcula el punto medio entre tl y br, el de tr y bl y luego vuelve a hacer su punto medio
    """
    p1 = middle_point(corners[0],corners[3])
    p2 = middle_point(corners[1],corners[2])
    MiddlePoint = middle_point(p1,p2)
    
    return MiddlePoint
    
def middle_point(point1, point2):
    """Calcula el punto medio de 2 puntos
    
    Args: 
        point1: Primer punto 
        point2: Segundo punto
        
    Return:
        OutPoint: Punto medio de point1 y point2
    """
    
    x = int((point1[0] + point2[0])/2)
    y = int((point1[1] + point2[1])/2)
    
    OutPoint = (x,y)
    
    return OutPoint

def manhattan(x,y):
   total = 0
   for i in range(len(x)):
     diff = x[i] - y[i]
     total = total + abs(diff)
   return total


def gaussian_smoothing(image, sigma, w_kernel):
    """ Blur and normalize input image.   
    
        Args:
            image: Input image to be binarized
            sigma: Standard deviation of the Gaussian distribution
            w_kernel: Kernel aperture size
                    
        Returns: 
            smoothed_norm: Blurred image
    """   
    # Write your code here!
    
    # Define 1D kernel
    s=sigma
    w=w_kernel
    kernel_1D = np.array([(1/(s*np.sqrt(2*np.pi)))*np.exp(-((pow(z,2))/(2*pow(s,2)))) for z in range(-w,w+1)])
    
    # Apply distributive property of convolution
    vertical_kernel = kernel_1D.reshape(2*w+1,1)
    horizontal_kernel = kernel_1D.reshape(1,2*w+1)   
    gaussian_kernel_2D = signal.convolve2d(vertical_kernel, horizontal_kernel)   
    
    # Blur image
    smoothed_img = cv2.filter2D(image,cv2.CV_16S,gaussian_kernel_2D)
    
    # Normalize to [0 254] values
    smoothed_norm = np.array(image.shape)
    smoothed_norm = cv2.normalize(smoothed_img ,smoothed_norm , 0, 255, cv2.NORM_MINMAX) # Leave the second argument as None
    
    return smoothed_norm

def binarize_kmeans(image,it):
    """ Binarize an image using k-means.   

        Args:
            image: Input image
            it: K-means iteration
    """    
    
    # Set random seed for centroids 
    cv2.setRNGSeed(124)
    
    # Flatten image
    flattened_img = image.reshape((-1,1))
    flattened_img = np.float32(flattened_img)
    
    #Set epsilon
    epsilon = 0.2
    
    # Estabish stopping criteria (either `it` iterations or moving less than `epsilon`)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, it, epsilon)
    
    # Set K parameter (2 for thresholding)
    K = 2
    
    # Call kmeans using random initial position for centroids
    _,label,center=cv2.kmeans(flattened_img,K,None,criteria,it,cv2.KMEANS_RANDOM_CENTERS)
    
    # Colour resultant labels
    center = np.uint8(center) # Get center coordinates as unsigned integers   
    ##print(center)
    flattened_img = center[label.flatten()] # Get the color (center) assigned to each pixel
    
    # Reshape vector image to original shape
    binarized = flattened_img.reshape((image.shape))
    
    return binarized


def main():
    """Solo funciona con los marcadores que tengas almacenados como imagenes en la carpeta Markers"""
    cap = cv2.VideoCapture(1) # Selecciona la entrada de video, 0 para la camara del pc y 1 para cualquier camara externa
    
    calib= calibracion()    # Obejto calibracion 
    CameraMatrix, dist, esquinas = calib.calibracion_cam()    
    print("Matriz de la camara: ", CameraMatrix)
    ##print("El tipo de la matriz es: ", type(CameraMatrix))
    print("Coeficiciente de Distorsion: ", dist)
    
    ##imgAug = cv2.imread("Markers/23.jpg")
    
    augDics = loadAugImages("Markers")
    while True:
        sccuuess, frame = cap.read()  # Toma imagenes de la camara 
        arucoFound = findArucoMarkers(frame) # devuelve [bbox,id]
        
        ## Loop through all the markers and augment each one
        if len(arucoFound[0])!=0:
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                print(bbox)
                if int(id) in augDics.keys():
                    frame = augmentAruco(bbox, id, frame, augDics[int(id)])
                    ##rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(arucoFound[id], 0.02, CameraMatrix, dist)
                    ##(rvec-tvec).any()
                    

        
        cv2.imshow("Image", frame)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()

