
import argparse
import configparser
import io
import os
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.layers import (Conv2D, Input, ZeroPadding2D, Add, UpSampling2D, MaxPooling2D, Concatenate)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model as plot

#Ce fichier va permettre la transformation du modéle de Darknet en une version Keras

#Utilisation de argpase afin de facilité l'utilisation des différent fichier qui vont être utilisé

pars = argparse.ArgumentParser(description='Convertion de Darknet vers Keras.') #donne la description  de ce que faire le fichier
pars.add_argument('config_path', help='Chemin du fichier Darknet cfg .')        # ajoute l'argument pour le fichier cfg contenant les blocs
pars.add_argument('weights_path', help='Chemin du fichier Darknet weights.')   # ajoute l'argument pour les poids
pars.add_argument('output_path', help='Chemin du fichier de sortie du model de Keras model.')  # ajoute l'argument de sortie
pars.add_argument('-p','--plot_model',help='Plot le modéle Keras créé et l enregistre comme image.',action='store_true')
pars.add_argument('-w','--weights_only',help='Enregistre en fichier poids de Keras au lieu du modéle.',action='store_true')


#fonction qui permet d'enregistrer toutes les valeurs du modéle et la création du modéle avec Keras
def main2(args):
    config_p = os.path.expanduser(args.config_path)
    # args.config_path permet d'aller chercher l'argument du cfg de pars et os.path.expanduser() permet de lire le nom du fichier  qui sera enregister dans config_p
    #identique pour le fichier des poids
    weights_p = os.path.expanduser(args.weights_path)
    assert config_p.endswith('.cfg'), '{} is not a .cfg file'.format(config_p) #permet de vérifier si le fichier de config est bien un .cfg
    assert weights_p.endswith('.weights'), '{} is not a .weights file'.format(weights_p)#permet de vérifier si le fichier de config est bien un .weights
    output_p = os.path.expanduser(args.output_path) # permet de lire le nom du fichier de sortie
    assert output_p.endswith('.h5'), 'output path {} is not a .h5 file'.format(output_p)#permet de vérifier si le fichier de sortie est bien un .h5
    output_r = os.path.splitext(output_p)[0] #permet d'enlever le .h5 à la fin

    print("Chargement des poids")
    fichier_weights = open(weights_p, 'rb') #permet l'ouverture du fichier weights
    M, m, revision= np.ndarray(shape=(3, ), dtype='int32', buffer=fichier_weights.read(12))

    if (M*10+m)>=2 and M<1000 and m<1000:
        seen = np.ndarray(shape=(1,), dtype='int64', buffer=fichier_weights.read(8))
    else:
        seen = np.ndarray(shape=(1,), dtype='int32', buffer=fichier_weights.read(4))
    print('Weights Header: ', M, m, revision, seen)

    print('Analyse de la configuration de Darknet.')
    config_uni = config_unique(config_p) #utilisation de la fonction config_unique
    cfg_pars = configparser.ConfigParser()
    cfg_pars.read_file(config_uni)  #lecture du fichier config_uni

    print('Création du model Keras.')
    input_couche = Input(shape=(None, None, 3)) #création d'un tenseur
    prev_couche = input_couche # enregistre la couche précédente
    all_couche = []  # permet d'enregistrer toutes les couches

    decay_weight = float(cfg_pars['net_0']['decay']) if 'net_0' in cfg_pars.sections() else 5e-4 #enregistre la valeur du decay qui se trouve dans le net
    counter = 0           #initialisation d'un compteur
    index_fin = []

    for section in cfg_pars.sections():                  #pour chaque bloc du fichier yolov3.cfg
        print('Parsing section {}'.format(section))
        if section.startswith('convolutional'):            #rentre quand c'est un block de convolution
            filters = int(cfg_pars[section]['filters'])  # enregistre le fitre qui le nombre de noyaux
            size = int(cfg_pars[section]['size'])        # size est la taille du noyau
            stride = int(cfg_pars[section]['stride'])    # pour savoir le déplacement du noyau à travers le canal d'entrée (x,y)
            pad = int(cfg_pars[section]['pad'])          # permet de savoir si il y a du padding ou pas
            activation = cfg_pars[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_pars[section] #enregistre si il y a du batch normalize, il applique une normalisation pour avoir de meilleur resultat

            padding = 'same' if pad == 1 and stride == 1 else 'valid' #enregistre same si pad et stride = 1 aussi non enregistre valid

            prev_couche_shape = K.int_shape(prev_couche) # donne les dimentions de prev_couche

            weights_shape = (size, size, prev_couche_shape[-1], filters) #enregistre les dimentions de weight
            darknet_w_shape = (filters, weights_shape[2], size, size)
            weights_size = np.product(weights_shape)

            print('conv2d', 'bn' if batch_normalize else '  ', activation, weights_shape)
            # permet d'afficher les la convolution , si il y a du batch normalize , le type d'activation et le poids (qui correspond au noyaux)

            conv_bias = np.ndarray(shape=(filters,),dtype='float32',buffer=fichier_weights.read(filters * 4))
            counter += filters #ajouter au counter la valeur du filtre

            if batch_normalize: # si il y a du batch normalize
                bn_weights = np.ndarray(shape=(3, filters),dtype='float32',buffer=fichier_weights.read(filters * 12))
                counter += 3 * filters

                bn_weight_list = [bn_weights[0], conv_bias, bn_weights[1], bn_weights[2] ] #création d'une liste avec chaque valeur

            conv_weights = np.ndarray(shape=darknet_w_shape,dtype='float32',buffer=fichier_weights.read(weights_size * 4))
            counter += weights_size

            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
            conv_weights = [conv_weights] if batch_normalize else [conv_weights, conv_bias]

            act_fn = None
            if activation == 'leaky':
                pass
            elif activation != 'linear':               # ne prends pas les activation différente de linear et leaky
                raise ValueError('Unknown activation function `{}` in section {}'.format(activation, section))

            # Create Conv2D layer
            if stride > 1:
                # Darknet uses left and top padding instead of 'same' mode
                prev_couche = ZeroPadding2D(((1, 0), (1, 0)))(prev_couche) #application de la fonction ZeroPadding2D
            conv_couche = (Conv2D(filters, (size, size),strides=(stride, stride),kernel_regularizer=l2(decay_weight),use_bias=not batch_normalize, weights=conv_weights,activation=act_fn,padding=padding))(prev_couche)

            if batch_normalize: # si il y a du batch normalize
                conv_couche = (BatchNormalization(weights=bn_weight_list))(conv_couche) # application da Batch normaliza
            prev_couche = conv_couche # enregistrement de la valeur

            if activation == 'linear':         # si c'est de type linear on ajout juste la couche précédente
                all_couche.append(prev_couche)
            elif activation == 'leaky':        # si c'est de type leaky
                act_couche = LeakyReLU(alpha=0.1)(prev_couche) # on ajoute un filtre de type leakyrelu sur la couche
                prev_couche = act_couche       # enregistrement de la couche
                all_couche.append(act_couche)  # ajout de la couche avec les modifications


        elif section.startswith('route'):          # si le bloc est de type route
            nbcouche = [int(i) for i in cfg_pars[section]['layers'].split(',')] # pour chaque valeur de layers
            couches = [all_couche[i] for i in nbcouche] #pour chaque valeur jusqu'a nbcouche on prend la valeur de all_couche qui lui correspond
            if len(couches) > 1:   #si la taille de couche est supérieur à 1
                print('Concatenating route layers:', couches)
                concatenate_couche = Concatenate()(couches)
                all_couche.append(concatenate_couche)
                prev_couche = concatenate_couche
            else:                                        # si le route est composé d'un seul élément
                skip_couche = couches[0]                 # la valeur de couche qu'on passe
                all_couche.append(skip_couche)           # ajout de la de la valeur au vecteur
                prev_couche = skip_couche                # enregistre la valeur

        elif section.startswith('maxpool'):         # si le bloc est de type maxpool
            size = int(cfg_pars[section]['size'])   #enregistre size
            stride = int(cfg_pars[section]['stride']) #enregistre stride
            all_couche.append(MaxPooling2D(pool_size=(size, size),strides=(stride, stride),padding='same')(prev_couche)) #application de la fonction Maxpooling et ajout au vecteur
            prev_couche = all_couche[-1]  # enregistre la derniére couche

        elif section.startswith('shortcut'):          # si le bloc est de type shortcut
            # le shortcut permet d'améliorer le réseau et passant des couches , ce sont des connexions de raccourcis
            index = int(cfg_pars[section]['from'])    # enregistre le nombre couche qu'on passe
            activation = cfg_pars[section]['activation'] # enregistre le type d'activation
            assert activation == 'linear', 'Only linear activation supported.' # si l'activation n'est pas de type linéaire
            all_couche.append(Add()([all_couche[index], prev_couche])) #ajoute la couche de type index et la couche précédente
            prev_couche = all_couche[-1] # enregistre la derniére couche

        elif section.startswith('upsample'):    #si c'est de type upsample ce qui va permettre d'augmenter les données
            stride = int(cfg_pars[section]['stride'])  #enregistre la valeur de stride
            assert stride == 2, 'Only stride=2 supported.'   # si stride n'est pas de valeur 2
            all_couche.append(UpSampling2D(stride)(prev_couche)) #application de la fonction Upsampling
            prev_couche = all_couche[-1] # on enregistre la derniére couche

        elif section.startswith('yolo'):       #si c'est de type yolo
            index_fin.append(len(all_couche)-1)   #ajout de l'index de fin au vecteur index_fin
            all_couche.append(None)              # ajout de la valeur "None" au vecteur
            prev_couche = all_couche[-1]        # on enregistre la derniére couche

        elif section.startswith('net'): # si c'est de type net on passe
            pass

        else:
            raise ValueError('Valeur de la section pas prise : {}'.format(section))

    #cette partie va créer le model et afficher les différents éléments du model
    if len(index_fin) == 0: index_fin.append(len(all_couche)-1) # si l'index de fin = 0 on ajoute la taille du vecteur
    model = Model(inputs=input_couche, outputs=[all_couche[i] for i in index_fin]) #création du model
    print(model.summary()) # permet d'afficher Total params:  Trainable params:  Non-trainable params:
    if args.weights_only:            # afficher le fichier d'enregistrement
        model.save_weights('{}'.format(output_p))
        print('Saved Keras weights to {}'.format(output_p))
    else:
        model.save('{}'.format(output_p))
        print('Saved Keras model to {}'.format(output_p))

    #permet de vérifier que tout les poids ont été utilisé
    remaining_weights = len(fichier_weights.read()) / 4
    fichier_weights.close() #fermeture du fichier
    print('Read {} of {} from Darknet weights.'.format(counter, counter +remaining_weights)) # affiche le nombre de poids utilisé par rapport au nombre de poids total
    if remaining_weights > 0:
        print('Attention: {} poids pas utilisé'.format(remaining_weights)) # affiche que tout les poids ne sont pas utilisé

    if args.plot_model:  #permet d'enregistrer le model sous forme d'image
        plot(model, to_file='{}.png'.format(output_r), show_shapes=True)
        print('Model enregistré à {}.png'.format(output_r))


#cette fonction permet de transformer le fichier .cfg pour pouvoir travailler plus facilement
def config_unique(config):
    counters = defaultdict(int) #création d'un dictionnaire qui va permettre de compter chaque fois qu'on rencontre "net" "convolution" ...

    output_stream = io.StringIO() #création d'un élément string pour enrigstrer les valeurs
    with open(config) as fin: #permet d'ouvrire le fichier config
        for ligne in fin:  # pour chauqe ligne
            if ligne.startswith('['):  # quand une ligne commence par "[" ce qui implique qu'on commence un bloc
                name = ligne.strip().strip('[]')  # on enléve les crochets autour du nom du bloc
                name2 = name + '_' + str(counters[name])  #enregistre le name et conteurs du name
                counters[name] += 1                   #ajoute 1 à la valeur de la clé "section"
                ligne = ligne.replace(name, name2)    # remplace dans la ligne la valeur name par name2
            output_stream.write(ligne)                # écrit la valeur de la ligne
    output_stream.seek(0)                             # remet le fichier à l'indice 0

    return output_stream  #retourne la valeur

if __name__ == '__main__': # va se lancer à chaque fois
    main2(pars.parse_args()) #applique la fonction main2