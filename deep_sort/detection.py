import numpy as np


class Detection(object):


    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float) #représente la box avec les coordonées x,y du point en haut à gauche , la largeur et la hauteur de la box
        self.confidence = float(confidence)          # score de confiance
        self.feature = np.asarray(feature, dtype=np.float32) #type d'objet

    def to_tlbr(self): # permet de transformer les données de la box en donnée des points en haut à gauche et en bas droite

        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self): #converti les données de la box en centre (x,y) et rapport de la largeur sur la hauteur

        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret