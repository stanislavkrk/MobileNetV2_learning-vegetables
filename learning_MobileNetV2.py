import os
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


def learning_model():

    '''Директорії'''
    TRAIN_DIR = "train"
    VAL_DIR = "val"
    MODEL_SAVE_PATH = "vegetable_model.h5"
    CLASS_INDICES_SAVE_PATH = "class_indices.json"


    '''Параметри'''
    IMG_SIZE = 224
    BATCH_SIZE = 8
    EPOCHS = 10


    '''
    Генератори даних із аугментацією.
    ImageDataGenerator — інструмент для автоматичної підготовки зображень для навчання нейронних мереж.
    Робимо аугментацію, для додаткової варіативності зображень. Це покращує впізнавання. Ми робимо:
    - масштабування значень пікселів до діапазону [0, 1] (замість оригінального [0, 255])
    - випадкове обертання зображення в межах 30 градусів (як у позитивний, так і в негативний бік)
    - випадкове горизонтальне зміщення зображення на 20% від його ширини. Імітує випадкове розташування об'єкта в кадрі по горизонталі.
    - випадкове вертикальне зміщення зображення на 20% від його висоти. Імітує випадкове розташування об'єкта в кадрі по вертикалі.
    - Випадкове застосування зрізу (shear transformation) з кутом зрізу до 20%. Це деформує зображення, наче воно 
      нахиляється в одну сторону, імітуючи різні перспективи.
    - Випадкове масштабування зображення (збільшення чи зменшення) в діапазоні до 20%. Це додає варіативності в розмірі об'єкта.
    - Випадкове горизонтальне віддзеркалення (фліп).
    - Задає спосіб заповнення нових пікселів, які з'являються в результаті трансформацій (обертання, зміщення, зрізу тощо).
      'nearest' означає, що нові пікселі будуть заповнюватися найближчими існуючими пікселями.
    '''
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )


    '''
    Для валідаційного набору масштабування значень пікселів до діапазону [0, 1] (замість оригінального [0, 255])
    '''
    val_datagen = ImageDataGenerator(rescale=1.0/255)


    '''
    Створюємо train+generator, який читає зображення із тренувальної директорії, робить їх обробку train_datagen, 
    подає їх партіями batch в модель. 
    Формат міток класів - 'categorical': Мітки повертаються у вигляді one-hot векторів (наприклад, [1, 0, 0] для "carrot").
    '''
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )


    '''
    Генератор для валідаційного набору.
    '''
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )


    '''
    Створення базової моделі MobileNetV2
    '''
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False  # Замороження базових шарів


    '''
    Додавання кастомних шарів. Створюємо модель нейронної мережі, використовуючи Sequential API, яка складається з:
    - базова модель - MobileNetV2, 
    - шари обробки даних після базової моделі, включаючи регуляризацію (Dropout), повнозв'язні шари (Dense) і функцію активації (softmax).
    Шари базової моделі зазвичай заморожені (не тренуються), щоб зберегти попередньо навчені ваги.
    
    GlobalAveragePooling2D() - замість розгортання всіх значень вихідного тензора в один великий вектор (як у Flatten()), 
    цей шар обчислює середнє значення для кожного каналу ознак. Це допомагає зменшити кількість параметрів і підвищити 
    загальну продуктивність.
    
    Dropout(0.5) - це регуляризаційний шар, який випадковим чином "відключає" 50% нейронів під час навчання. Його основна мета — 
    уникнути перенавчання, додаючи шум у навчальний процес, змушуючи модель покладатися на всі нейрони, а не на певну підмножину.
    
    Dense(128, activation='relu') - повнозв'язний шар із 128 нейронами, activation='relu' — функція активації, 
    яка залишає лише позитивні значення (ефективна для глибоких мереж). Цей шар використовується для навчання високорівневих 
    взаємозв’язків між ознаками.
    
    Другий Dropout(0.5) - відключає 50% нейронів перед передачею виходу в останній шар. Це додаткова регуляризація, 
    яка допомагає уникнути перенавчання на рівні повнозв'язних шарів.
    
    Dense(train_generator.num_classes, activation='softmax') - остаточний шар класифікації. Кількість виходів 
    (train_generator.num_classes) відповідає кількості класів у датасеті. activation='softmax' — функція активації 
    для багатокласової класифікації, яка повертає ймовірності для кожного класу.
    '''
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),  # Регуляризація для уникнення перенавчання
        Dense(128, activation='relu'),
        Dropout(0.5),  # Ще один шар Dropout
        Dense(train_generator.num_classes, activation='softmax')
    ])


    '''
    Функція model.compile конфігурує модель для навчання, визначаючи:
    Оптимізатор — як оновлюються ваги моделі, optimizer=Adam(learning_rate=0.0001). Adam (Adaptive Moment Estimation) -
    популярний оптимізатор, який автоматично налаштовує швидкість навчання для кожного параметра моделі. 
    Комбінує переваги Momentum і RMSProp.
    '''
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    '''
    Використання EarlyStopping для запобігання перенавчанню
    '''
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


    '''
    Навчання моделі
    '''
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=[early_stopping]
    )


    '''
    Розмороження базової моделі для fine-tuning:
    base_model.trainable = True - це дозволяє навчання шарів у базовій моделі (base_model).
    for layer in base_model.layers[:-20] - ітерація через шари базової моделі, окрім останніх 20,
    [:-20] вибирає всі шари, крім останніх 20. Решту заморожує.
    Тонке налаштування (Fine-tuning):
    останні кілька шарів базової моделі відповідають за високорівневі ознаки, які більше залежать від специфіки нового датасету.
    Розмороження останніх шарів дозволяє моделі адаптуватись до нових даних, не змінюючи низькорівневих ознак.
    '''
    base_model.trainable = True
    for layer in base_model.layers[:-20]:  # Замороження всіх, крім останніх 20 шарів
        layer.trainable = False


    '''
    Цей код компілює модель з оновленими параметрами для навчання.
    optimizer=Adam(learning_rate=1e-5) - встановлює оптимізатор Adam із дуже низькою швидкістю навчання (1e-5 або 0.00001).
    Це дозволяє робити дуже повільні, але точні оновлення вагів.
    
    loss='categorical_crossentropy' - визначає функцію втрат для багатокласової класифікації.
    Використовується, коли мітки класів представлені у вигляді one-hot векторів (наприклад, [1, 0, 0]).
    Встановлює метрику точності (accuracy) для моніторингу під час навчання та валідації.
    '''
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    '''Додаткове навчання'''
    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=5,  # Декілька епох для уточнення
        callbacks=[early_stopping]
    )


    '''Збереження моделі'''
    model.save(MODEL_SAVE_PATH)


    '''Збереження словника класів'''
    class_indices = train_generator.class_indices
    with open(CLASS_INDICES_SAVE_PATH, "w") as f:
        json.dump(class_indices, f)

    print(f"Модель і словник класів збережено.")


if __name__ == '__main__':

    learning_model()