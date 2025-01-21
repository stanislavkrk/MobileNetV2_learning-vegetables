import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def video_recognizer():

    '''Шляхи до файлів'''
    MODEL_PATH = "vegetable_model.h5"
    CLASS_INDICES_PATH = "class_indices.json"
    CLUSTER_OUTPUT_PATH = "vegetable_clusters.json"

    '''Параметри'''
    IMG_SIZE = 224  # Розмір зображень
    CONFIDENCE_THRESHOLD = 0.7  # Поріг впевненості

    '''Завантаження навченої моделі'''
    model = load_model(MODEL_PATH)

    '''Завантаження словника класів'''
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    class_labels = {v: k for k, v in class_indices.items()}

    '''Завантаження кластерів'''
    with open(CLUSTER_OUTPUT_PATH, "r") as f:
        clusters = json.load(f)

    '''
    Функція для передбачення класу та кластера.
    cv2.resize(frame, (IMG_SIZE, IMG_SIZE)) - масштабує вхідний кадр до розміру.
    preprocess_input(resized_img) - нормалізує зображення, використовуючи ту ж функцію, що й під час тренування.
    Це забезпечує правильну обробку пікселів перед передачею моделі.
    np.expand_dims(processed_img, axis=0) - додає вимір "батчу" (розмірність 1), оскільки модель очікує вхід у форматі 
    (batch_size, IMG_SIZE, IMG_SIZE, 3).
    model.predict(input_data)[0] - подає оброблене зображення в модель для передбачення. Повертає ймовірності для всіх класів 
    (вихід softmax). [0] витягує перший (і єдиний) елемент батчу.
    confidence = np.max(prediction) - знаходить найбільшу ймовірність серед усіх класів. Це показник впевненості моделі в передбаченні.
    Якщо впевненість (ймовірність) менша за порогове значення CONFIDENCE_THRESHOLD, функція повертає:
        None для назви класу.
        None для кластера.
    np.argmax(prediction) - визначає індекс класу з найвищою ймовірністю.
    class_labels[predicted_class_index] - отримує назву класу за індексом.
    clusters[predicted_class_name] - за назвою класу визначає, до якого кластера належить передбачений об'єкт.
    '''
    def predict_class_and_cluster(frame):
        resized_img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        processed_img = preprocess_input(resized_img)
        input_data = np.expand_dims(processed_img, axis=0)

        prediction = model.predict(input_data)[0]  # Отримання ймовірностей
        confidence = np.max(prediction)  # Найвища ймовірність
        if confidence < CONFIDENCE_THRESHOLD:
            return None, None, confidence  # Невпевнене передбачення

        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_labels[predicted_class_index]
        predicted_cluster = clusters[predicted_class_name]

        return predicted_class_name, predicted_cluster, confidence

    '''Захоплення відеопотоку'''
    video_path = "vegetables_video.mp4"  # Шлях до відео
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Не вдалося відкрити відео.")
        exit()

    '''
    Цикл обробляє відеопотік кадр за кадром. Для кожного кадру виконується:
    Розпізнавання класу об’єкта та його кластера за допомогою функції predict_class_and_cluster.
    Якщо розпізнавання впевнене, кадру додається текст із назвою класу, кластером і рівнем впевненості, а також малюється рамка.
    Кадр виводиться у вікні OpenCV.
    Відео можна зупинити натисканням клавіші q.
    
    ret, frame = cap.read() - зчитує наступний кадр із відеооб'єкта cap. ret — прапорець, що показує, чи успішно було 
    прочитано кадр, frame — сам кадр з відео.
    if not ret: break - якщо відео закінчилось або не вдалось прочитати кадр, цикл завершується.
    Викликається функція predict_class_and_cluster, яка повертає:
        class_name — назва передбаченого класу (наприклад, "tomato").
        cluster — номер кластера (наприклад, 1, 2, 3).
        confidence — рівень впевненості в передбаченні (від 0 до 1).
    Якщо функція predict_class_and_cluster повернула значення (не None), тобто впевненість у передбаченні достатня.
    Якщо виникає помилка під час обробки кадру (наприклад, через некоректний вхід), програма не завершується, а виводить текст помилки.
    cap.release() - закриває об'єкт відео, звільняючи ресурси.
    cv2.destroyAllWindows() - закриває всі вікна OpenCV.
    '''
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Передбачення для кожного кадру
        try:
            class_name, cluster, confidence = predict_class_and_cluster(frame)

            if class_name:  # Якщо впевненість достатня
                # Додавання тексту та рамки на кадр
                text = f"{class_name}, Cluster: {cluster} ({confidence:.2f})"
                cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (50, 50), (frame.shape[1] - 50, frame.shape[0] - 50), (0, 255, 0), 2)
        except Exception as e:
            print(f"Помилка обробки кадру: {e}")

        # Показ відео
        cv2.imshow("Vegetables recognizer", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    video_recognizer()