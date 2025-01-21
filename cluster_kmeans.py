import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.cluster import KMeans
import cv2

def cluster_kmeans():

    '''Шляхи до файлів'''
    MODEL_PATH = "vegetable_model.h5"
    CLASS_INDICES_PATH = "class_indices.json"
    CLUSTER_OUTPUT_PATH = "vegetable_clusters.json"


    '''Параметри'''
    IMG_SIZE = 224  # Розмір зображень
    NUM_CLUSTERS = 5  # Кількість кластерів
    BATCH_SIZE = 32  # Розмір батчу для обробки


    '''Завантаження навченої моделі'''
    model = load_model(MODEL_PATH)


    '''Завантаження словника класів'''
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)


    '''Інвертування словника (для отримання назв класів)'''
    class_labels = {v: k for k, v in class_indices.items()}


    '''Функція для витягування ознак батчами'''
    def extract_features_batch(image_paths, batch_size=BATCH_SIZE):
        features = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []

            for img_path in batch_paths:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = preprocess_input(img)
                batch_images.append(img)

            batch_images = np.array(batch_images)
            batch_features = model.predict(batch_images)
            features.extend(batch_features)

        return np.array(features)


    '''Збір шляхів до зображень із тренувального датасету'''
    DATA_DIR = "train"
    image_paths = []
    image_labels = []  # Для зберігання відповідних назв класів

    for label in os.listdir(DATA_DIR):
        class_dir = os.path.join(DATA_DIR, label)
        if os.path.isdir(class_dir):  # Перевіряємо, чи це папка
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                image_paths.append(img_path)
                image_labels.append(label)

    '''Витягування ознак батчами'''
    print("Витягування ознак із зображень...")
    features = extract_features_batch(image_paths)


    '''Кластеризація за допомогою K-Means'''
    print("Виконується кластеризація...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
    clusters = kmeans.fit_predict(features)


    '''Формування результатів кластеризації'''
    cluster_results = {}
    for i, label in enumerate(image_labels):
        cluster_results[label] = int(clusters[i])


    '''Запис у файл'''
    with open(CLUSTER_OUTPUT_PATH, "w") as f:
        json.dump(cluster_results, f)

    print(f"Кластеризацію завершено. Результати збережено у {CLUSTER_OUTPUT_PATH}.")

if __name__ == '__main__':

    cluster_kmeans()