from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compo

def Convol2D_Darknet(*args, **kwargs): #"*" et "**" Permet de "dépaqueter" listes, dico,etc. en arguments unitaires

    #définition des paramètres Darknet pour la convolution 2D
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)} #fonction l2 = Somme des carrés de tous les poids --> oblige les poids à être <<< mais != 0
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs) #remet à jour les paramètres
    return Conv2D(*args, **darknet_conv_kwargs) #Keras Conv2D est une couche de convolution 2D, cette couche crée un noyau de convolution (couches d'entrées) qui aide à produire un tenseur de sorties

#Utilisation de la convolution 2D de darknet avec la "batchnomarlization" et "LeakyRelu"
def Convol2D_Darknet_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False} #paramètre entrainable supplémentaire mais non-obligatoire : ici = False
    no_bias_kwargs.update(kwargs) #on remet à jour
    return compo(Convol2D_Darknet(*args, **no_bias_kwargs), BatchNormalization(), LeakyReLU(alpha=0.1))

#Darknet avec 52 couches de convolutions 2D
def body_Darknet(y):

    y = Convol2D_Darknet_BN_Leaky(32, (3, 3))(y)# SIDI ?
    #on applique 5 fois la fonction body_resblock(a, filters, blocks)
    y = body_resblock(y, 64, 1)
    y = body_resblock(y, 128, 2)
    y = body_resblock(y, 256, 8)
    y = body_resblock(y, 512, 8)
    y = body_resblock(y, 1024, 4)
    return y

def body_resblock(a, filters, blocks): #Définition d'une série de "resblocks" commençant par un sous-échantillonnage Convolution2D

    #Darknet utilise "left et top padding" à la place du mode 'same'
    a = ZeroPadding2D(((1, 0), (1, 0)))(a) #ajoute des lignes en haut à gauche ou bien à droite
    a = Convol2D_Darknet_BN_Leaky(filters, (3, 3), strides=(2, 2))(a) # SIDI ?
    for i in range(blocks):
        b = compo(Convol2D_Darknet_BN_Leaky(filters // 2, (1, 1)), Convol2D_Darknet_BN_Leaky(filters, (3, 3)))(a) # SIDI ?
        a = Add()([a, b]) #ajoute l'élement uniquement si il n'est pas encore présent
    return a

def dernieres_couches(x, num_filters, out_filters):
    #6 couches Conv2D_BN_Leaky suivies d'une couche Conv2D_linear
    x = compo(
        Convol2D_Darknet_BN_Leaky(num_filters, (1, 1)),
        Convol2D_Darknet_BN_Leaky(num_filters*2, (3, 3)),
        Convol2D_Darknet_BN_Leaky(num_filters, (1, 1)),
        Convol2D_Darknet_BN_Leaky(num_filters*2, (3, 3)),
        Convol2D_Darknet_BN_Leaky(num_filters, (1, 1)))(x)
    y = compo(
        Convol2D_Darknet_BN_Leaky(num_filters*2, (3, 3)),
        Convol2D_Darknet(out_filters, (1, 1)))(x)

    return x, y


def body_yolo(inputs, num_anchors, num_classes):
    #création d'un modéle de Yolo-v3 en utilisant un réseau de neurones convolutif
    darknet = Model(inputs, body_Darknet(inputs))
    x, y1 = dernieres_couches(darknet.output, 512, num_anchors*(num_classes+5))

    x = compo(
        Convol2D_Darknet_BN_Leaky(256, (1, 1)), #1 couche Conv2D_BN_Leaky
        UpSampling2D(2))(x)#Répète les lignes de données de "taille [0]" et "taille [1]" respectivement
    x = Concatenate()[x, darknet.layers[152].outpout] #permet de regrouper les éléments ensemble
    x, y2 = dernieres_couches(x, 256, num_anchors*(num_classes+5))

    x = compo(
        Convol2D_Darknet_BN_Leaky(128, (1, 1)), #1 couche Conv2D_BN_Leaky
        UpSampling2D(2))(x) #Répète les lignes de données de "taille [0]" et "taille [1]" respectivement
    x = Concatenate()[x, darknet.layers[92].outpout]  # permet de regrouper les éléments ensemble
    x, y3 = dernieres_couches(x, 128, num_anchors * (num_classes + 5))

    return Model(inputs, [y1, y2, y3])

def tiny_body_yolo(inputs, num_anchors, num_classes):
    #Création d'un PETIT modéle de Yolo-v3 en utilisant un réseau de neurones convolutif

    #5 couches Conv2D_BN_Leaky
    x1 = compo(
        Convol2D_Darknet_BN_Leaky(16, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'), #Opération de regroupement qui calcule la valeur maximale de chaque paquet de chaque carte d'entités
        Convol2D_Darknet_BN_Leaky(32, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        Convol2D_Darknet_BN_Leaky(64, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        Convol2D_Darknet_BN_Leaky(128, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        Convol2D_Darknet_BN_Leaky(256, (3, 3)))(inputs)

    #3 couches Conv2D_BN_Leaky
    x2 = compo(
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        Convol2D_Darknet_BN_Leaky(512, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        Convol2D_Darknet_BN_Leaky(1024, (3, 3)),
        Convol2D_Darknet_BN_Leaky(256, (1, 1)))(x1)

    #1 couche Conv2D_BN_Leaky suivie d'une couche Conv2D_linear
    y1 = compo(
        Convol2D_Darknet_BN_Leaky(512, (3, 3)),
        Convol2D_Darknet(num_anchors*(num_classes+5), (1, 1)))(x2)

    #1 couches Conv2D_BN_Leaky suivie d'un UpSampling
    x2 = compo(
        Convol2D_Darknet_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x2)

    #Concaténation, 1 couches Conv2D_BN_Leaky suivie d'une couche Conv2D_linear
    y2 = compo(
        Concatenate(),
        Convol2D_Darknet_BN_Leaky(256, (3, 3)),
        Convol2D_Darknet(num_anchors*(num_classes+5), (1, 1)))([x2, x1])

    return Model(inputs, [y1, y2])

##################################################################

def yolo_eval(yolo_sortie,anchors,num_classes, image_shape, max_boxes=20,  score_threshold=.6, iou_threshold=.5):
    #Évaluer le modèle  YOLO sur une entrée  donnée  et renvoyer  des boîtes  filtrées.
    num_couches = len(yolo_sortie) #prends le nombre de couceh
    mask_anchor = [[6,7,8], [3,4,5], [0,1,2]] if num_couches==3 else [[3,4,5], [1,2,3]] #donne des valeurs par défaut
    input_forme = K.shape(yolo_sortie[0])[1:3] * 32
    boxes = [] #création d'un vecteur pour les boxes
    scores_box = [] #création d'un vecteur pour les scores des boxes

    for l in range(num_couches):
        boxes_2, box_scores_2 = yolo_boxes_and_scores(yolo_sortie[l], anchors[mask_anchor[l]], num_classes, input_forme, image_shape) #utilisation de la fonction "yolo_boxes_and_scores"
        boxes.append(boxes_2)
        scores_box.append(box_scores_2)
    boxes = K.concatenate(boxes, axis=0) #regroupe les éléments de la même boxes ensemble (ex: si on a un élément de forme (2,2,5) on va obtenir un élément de forme (4,5))
    box_scores = K.concatenate(scores_box, axis=0)

    mask = box_scores >= score_threshold #création d'un vecteur avec des boolean True quand cest plus grand ou égale et false dans l'autre cas
    max_boxes_tensor = K.constant(max_boxes, dtype='int32') #création d'un tenseur avec l'élément max boxes
    #créations de nouveaux vecteur
    boxes_3 = []
    scores_3 = []
    classes_3 = []

    for i in range(num_classes):  #boucler sur toutes les classes
        class_boxes = tf.boolean_mask(boxes, mask[:, i]) #gardes que les valeurs de boxes quand la valeur de mask vaut true
        class_box_scores = tf.boolean_mask(box_scores[:, i], mask[:, i]) #gardes que les scores de boxes quand la valeur de mask vaut true
        # calcul chevauchement  (IOU) entre les boxs qui dépasse le seuil avec les boîtes précédemment sélectionnées
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index) #permet de garder que les boxs qui ne se chevauche pas trop
        class_box_scores = K.gather(class_box_scores, nms_index)  #permet de garder que les scores des boxs qui ne se chevauche pas trop
        classes = K.ones_like(class_box_scores, 'int32') * i #transforme class_box_scores avec que des 1 et multiplie par i pour avoir le numéro de la classe
        #ajouter les valeurs aux vecteur créer précédement
        boxes_3.append(class_boxes)
        scores_3.append(class_box_scores)
        classes_3.append(classes)
    boxes_3 = K.concatenate(boxes_3, axis=0)
    scores_3 = K.concatenate(scores_3, axis=0)
    classes_3 = K.concatenate(classes_3, axis=0)

    return boxes_3, scores_3, classes_3


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,anchors, num_classes, input_shape) #utilisation de la fonction yolo_head
    #box_xy = position en x,y
    #box_wh = largeur et hauteur de la box

    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape) #utilisation de la fonction yolo_correct_boxes
    boxes = K.reshape(boxes, [-1, 4])                                    #redimentionne les éléments de boxes
    box_scores = box_confidence * box_class_probs                        #permet de calculer le score
    box_scores = K.reshape(box_scores, [-1, num_classes])                #attribue les scores au classe
    return boxes, box_scores


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    # Conversion des entités de couche finale en paramètres de boîte englobante.
    num_anchors = len(anchors)

    # Remodelation en batch, hauteur, largeur, num_anchors, box_params
    anchors_tenseur = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors,
                                                      2])  # K.reshape() remodèlera non seulement le tableau, mais convertira également
    # le tableau en une structure tensorielle à l'aide du "backend"
    forme_grille = K.shape(feats)[1:3]  # Hauteur: 1; Largeur: 3
    grille_y = K.tile(K.reshape(K.arange(0, stop=forme_grille[0]), [-1, 1, 1, 1]),
                      # K.tile() --> Crée un nouveau tenseur en répliquant plusieurs fois la variable d'entrée.
                      [1, forme_grille[1], 1,
                       1])  # La i'ième dimension du tenseur de sortie a une entrée; K.arrange() --> tri
    grille_x = K.tile(K.reshape(K.arange(0, stop=forme_grille[1]), [1, -1, 1, 1]),
                      # K.tile() --> Crée un nouveau tenseur en répliquant plusieurs fois la variable d'entrée.
                      [forme_grille[0], 1, 1, 1])
    grille = K.concatenate([grille_x, grille_y])  # concaténation des deux grilles créées
    grille = K.cast(grille, K.dtype(feats))  # K.cast() --> conversion d'un tenseur dans un certain type

    feats = K.reshape(
        feats, [-1, forme_grille[0], forme_grille[1], num_anchors, num_classes + 5]
    )

    # Ajuster les préditions à chaque point de grille spatiale et à la taille de l'anchors
    box_xy = (K.sigmoid(feats[..., :2]) + grille) / K.cast(forme_grille[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tenseur / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:  # calc_loss ? M.SIDI ?
        return grille, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    # obtention des boxes corrigées
    box_yx = box_xy[..., ::-1]  # "::-1" --> parcours de la fin jusqu'au début (inverse)
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))  # On prend la forme d'entrée et on la "cast"
    image_shape = K.cast(image_shape, K.dtype(box_yx))  # On prend la forme de l'image et on la "cast"
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))  # round --> fonction échelon en français
    decalage = (input_shape - new_shape) / 2. / input_shape  # opération?? M.SIDI
    echelle = input_shape / new_shape
    box_yx = (box_yx - decalage) * echelle
    box_hw *= echelle

    min_box = box_yx - (box_hw / 2.)
    max_box = box_yx + (box_hw / 2.)
    boxes = K.concatenate(
        [
            min_box[..., 0:1],  # y minimum
            min_box[..., 1:2],  # x minimum
            max_box[..., 0:1],  # y maximum
            max_box[..., 1:2],  # x maximum
        ]
    )

    # Redimension des boîtes à la forme de l'image d'origine
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes