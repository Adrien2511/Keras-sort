from __future__ import division, print_function, absolute_import
import os
import datetime
from timeit import time
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from keras import backend

backend.clear_session()


chemin_model = "model_data/yolo.h5"  # on encode les chemin d'accès au modèle et à la vidéo "test"




classes = "model_dat/coco_classes.txt"  # on se base sur le modèle "coco" pour nos différentes classes

# enregistrement des cadres ("anchors") --> contour des box
# choix du modèle

anchors_yolo = "model_data/yolo_anchors.txt" # chemin pour les anchors


# enregistrement du modèle
yolo = YOLO(classes_path1=classes, anchors_path1=anchors_yolo, model_path1=chemin_model)

pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3),dtype="uint8")

def main(yolo):

    start = time.time()
    #Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 0.3 #donne la superposition maximale des boxs


    #deep_sort
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1) #utilisation de la fonction create_box_encoder du fichier generate_detection

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)  # création d'un objet de la classe NearestNeighborDistanceMetric du fichier nn_matching
    tracker = Tracker(metric) #création d'un élément de la class Tracker du fichier tracker

    writeVideo_flag = True

    video_file="videoface2.avi"  #chemin pour la vidéo
    video_capture = cv2.VideoCapture(video_file)  #ouverture de la vidéo
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # initialisation de la largeur de la video : 640px
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)


    # Define the codec and create VideoWriter object
    w = int(video_capture.get(3)) #prise de la largeur de la vidéo
    h = int(video_capture.get(4)) #prise de la hauteur de la vidéo
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #out = cv2.VideoWriter('sortie4.AVI', fourcc, 10, (w,h)) #création d'un élément pour pouvoir enregistrer

    frame_index = -1

    fps = 0.0  # création d'une variable pour calculer les fps
    totalframe = 0 #création d'une variable pour pouvoir n'utiliser que certaine frame


    while True:

            ret, frame = video_capture.read()  # frame shape 640*480*3

            if ret != True: #vérifier si la vidéo est bien ouverte
             break

        #if totalframe%3 == 0: #utilisation d'un modulo pour n'utiliser que certaine frame

            t1 = time.time()

            #image = Image.fromarray(frame)
            frame=cv2.resize(frame,(480,360))

            image = Image.fromarray(frame[...,::-1]) #transforme l'iamge en noir et blanc
            boxs,class_names, return_score = yolo.image_detection(image) #renvoi les coordonées de la box , la class et le score
            features = encoder(frame,boxs)
            # score to 1.0 here).
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections]) #enregistre les boxes
            scores = np.array([d.confidence for d in detections]) #enregistre les scores
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores) #retourne les indices des boxs garder
            detections = [detections[i] for i in indices] #garde que les éléments qui ont leur indice dans le vecteurs indice

            # Call the tracker
            tracker.predict() #appelle de la fonction predict de la classe tracker
            tracker.update(detections)

            i = int(0)
            indexIDs = [] # création d'un vecteur pour enrgistrer les indices
            c = []
            boxes = []


            for track in tracker.tracks: #appliquer pour chaque élément qu'on traque
                if not track.is_confirmed() or track.time_since_update > 1: # si pas de détection
                    continue

                indexIDs.append(int(track.track_id)) #enregistrement de l'indice

                bbox = track.to_tlbr()
                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]] # enregistre une couleur

                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3) #création du cadre
                cv2.putText(frame,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),2) #écriture  de l'id sur la box
                if len(class_names) > 0:
                   class_name = class_names[0]
                   cv2.putText(frame, str(class_names[0]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2) #écriture du nom de la classe

                i += 1
                #bbox_center_point(x,y)
                center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2)) # calcul le centre de la box
                #track_id[center]
                pts[track.track_id].append(center)
                thickness = 5 # la taille de l'élément au centre
                #center point
                cv2.circle(frame,  (center), 1, color, thickness) #crée l'élément au centre du cadre

            #draw motion path
                for j in range(1, len(pts[track.track_id])):
                    if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                       continue
                    thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                    #cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)
                    #cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)


            #cv2.putText(frame, "Total Object Counter: "+str(count),(int(20), int(120)),0, 5e-3 * 200, (0,255,0),2)
            #cv2.putText(frame, "Current Object Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
            #cv2.putText(frame, "FPS: %f"%(fps),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),3)
            cv2.namedWindow("YOLO3_Deep_SORT", 0); #donne le nom de la fenêtre
            cv2.resizeWindow('YOLO3_Deep_SORT', 640, 360); #donne la taille de la fenêtre
            cv2.imshow('YOLO3_Deep_SORT',frame) #affiche le resultat



            #save a frame
            #out.write(frame) #enregistre la frame
            frame_index = frame_index + 1 #augmente le nombre de frame

            fps  = ( fps + (1./(time.time()-t1)) ) / 2 #calcul des fps


            # Press Q to stop!
            totalframe += 1 #augmente le compteur de frame
            if cv2.waitKey(1) & 0xFF == ord('q'):  #permet de ferme la fenêtre
                break
    print(" ")
    print("[Finish]")
    end = time.time()


    video_capture.release()


    # if writeVideo_flag:
        #out.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())