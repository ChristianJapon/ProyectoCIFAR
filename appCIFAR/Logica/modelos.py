from django.urls import reverse
import numpy as np
from keras.preprocessing import image
import pickle
import keras
from sklearn.preprocessing import LabelEncoder
import os


class ModeloCIFAR():


    def __init__(self, ruta_modelo_cnn, ruta_modelo_svm):
        self.le = LabelEncoder()
        # Definir nombres de etiquetas
        self.label_names = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
            'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
            'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
            'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
            'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
            'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
            'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]
        self.modelo_cnn, self.modelo_svm = self.cargarModelo(ruta_modelo_cnn, ruta_modelo_svm)

    def cargarCNN(self, path):
        # Aquí deberías tener la lógica para cargar el modelo CNN
        # Puedes usar keras.models.load_model o tu propio método de carga
        # Ejemplo:
        modelo_cnn = keras.models.load_model(path)
        return modelo_cnn

    def cargarSVM(self, path):
        # Asegúrate de usar la misma versión de pickle que usaste para guardar el modelo
        with open(path, 'rb') as file:
            modelo_svm = pickle.load(file)
        return modelo_svm

    def cargarModelo(self, ruta_modelo_cnn, ruta_modelo_svm):   
        print(f"Intentando cargar el modelo CNN desde: {ruta_modelo_cnn}")
        modelo_cnn = self.cargarCNN(ruta_modelo_cnn)

        print(f"Intentando cargar el modelo SVM desde: {ruta_modelo_svm}")
        modelo_svm = self.cargarSVM(ruta_modelo_svm)

        print("Se están cargando los modelos")
        return modelo_cnn, modelo_svm

    def preprocessCNN(self, img_path):
        img = image.load_img(img_path, target_size=(32, 32))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        print("preprocesamientoCNN correcto")
        return img_array

    def preprocessSVM(self, img_path):
        img = image.load_img(img_path, target_size=(32, 32))
        img_array = image.img_to_array(img)
        img_array = img_array.flatten() / 255.0
        img_encoded = img_array.flatten().reshape(1, -1)  # Corregir la forma del array
        print("preprocesamientoSVM correcto")
        return img_encoded

    def predecirImagen(self, rutaImagen):
        img_cnn = self.preprocessCNN(rutaImagen)
        img_svm = self.preprocessSVM(rutaImagen)

        pred_cnn = self.modelo_cnn.predict(img_cnn)
        pred_svm = self.modelo_svm.predict(img_svm)

        # Obtener el índice de la clase predicha
        idx_cnn = pred_cnn.argmax()
        idx_svm = pred_svm[0]

        nombre_clase_cnn = self.label_names[idx_cnn]
        nombre_clase_svm = self.label_names[idx_svm]

        resultado_final = {
            "nombre_clase_cnn": nombre_clase_cnn,
            "nombre_clase_svm": nombre_clase_svm,
            "image": rutaImagen  # Cambiado para devolver la ruta de la imagen
        }
        return resultado_final
