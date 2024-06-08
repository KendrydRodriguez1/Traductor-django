import os
import cv2

ROOT_PATH = os.getcwd() #por akguna razon esyto te lleva a la carpeta madre
print("-----------------------")
print(ROOT_PATH)
print("-----------------------")
#esto crea la direccion de donde estara la carpeta frame_actions donde estarn la fotos con su nombre 
FRAME_ACTIONS_PATH = os.path.join(ROOT_PATH, "frame_actions") 
print("-----------------------")
print(FRAME_ACTIONS_PATH)
print("-----------------------")
#lo mismo  para una carpeta data y models 
DATA_PATH = os.path.join(ROOT_PATH, "data") #aqui estaran los h5
MODELS_PATH = os.path.join(ROOT_PATH, "models") #aqui estaran el keras

MAX_LENGTH_FRAMES = 15
LENGTH_KEYPOINTS = 1662
MIN_LENGTH_FRAMES = 5
MODEL_NAME = f"actions_{MAX_LENGTH_FRAMES}.keras"

# SHOW IMAGE PARAMETERS
FONT = cv2.FONT_HERSHEY_PLAIN #el tipo de letra que se mostraracuando abra la camara
FONT_SIZE = 1.5
FONT_POS = (5, 30) #posicion de la palabra 
