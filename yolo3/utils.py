from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

def compo(*funcs):

    if funcs: #si ce n'est pas vide
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    #La fonction reduction  permet d'appliquer une fonction au paramètres f et g  de "funcs".
    else:
        raise ValueError('La composition de la séquence vide n est pas prise en charge.') #return une valeur d'erreur si il y a pas de funcs


def letterbox_image(img, size):

    w_initiale, h_initiale =img.size # on prends les dimentions de l'image de base
    w, h = size #on prends les valeurs de l'attribut size
    taille = min(w/w_initiale, h/h_initiale) #prend le min entre le rapport de largeur et de hauteur

    new_w = int(taille*w_initiale) #calcul de la nouvelle largeur
    new_h = int(taille*h_initiale) #calcul de la nouvelle hauteur

    img=img.resize((new_w,new_h), Image.BICUBIC) #redimentionne l'image
    new_img = Image.new('RGB', size, (128,128,128)) #création d'une nouvelle image
    new_img.paste(img, ((w-new_w)//2, (h-new_h)//2)) #paste permet d'intégrer l'ancienne  sur la nouvelle en donnant la position

    return new_img
"""
def random_(x=0, y=1):
    return np.random.rand()*(y-x) + x #permet d'avoir une valeur aléatoire entre x et y

def data_reandom(annotation, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):

    ligne = annotation.split() #on divise les annotations
    img = Image.open(ligne[0])
    w1, h1 = img.size #prise de la hauteur et de la largeur de l'image
    h, w = input_shape
    box = np.array([np.array(list(map(int,box.split(',')))) for box in ligne[1:]])

    if not random: #si le random n'est pas activé
        # redimensionner l'image
        scale = min(w/w1, h/h1) #calcul du rapport entre les hauteurs et largeurs et prendre le minimum
        w2 = int(w1*scale)
        h2 = int(h1 * scale)
        dx = (w - w2) // 2
        dy = (h - h2) // 2
        data_img = 0
        if proc_img:
            img = img.resize((w2, h2), Image.BICUBIC) # redimensionne l'image
            img_new = Image.new('RGB', (w,h), (128,128,128))
            img_new.paste(img, (dx, dy))
            data_img = np.array(img_new)/255.

        data_box = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            data_box[:len(box)] = box

        return data_img, data_box

    ar = w/h *random_(1-jitter,1+jitter)/random_(1-jitter,1+jitter)
    scale = random_(.25, 2)
    if ar < 1:
        h2 = int(scale*h)  #calcul de la hauteur
        w2 = int(h2*ar)    #calcul de la largeur
    else:
        w2 = int(scale*w)  #calcul de la hauteur
        h2 = int(w2/ar)    #calcul de la largeur
    img = img.resize((w2,h2), Image.BICUBIC) # redimensionne l'image
"""




