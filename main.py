import learning_MobileNetV2 as learn
import cluster_kmeans as cluster
import pytube_download as download
import  video_recognizer as recognizer

def choose_main():
    print('Оберіть задачу для виконання:'
          '\n1 - Навчаємо модель MobileNetV2 на підготовленому датасеті із заданими параметрами'
          '\n2 - Виконуємо кластеризацію зображень на базі вже навченої моделі'
          '\n3 - Завантажити відео для роспізнавання з YouTube'
          '\n4 - Виконуємо роспізнавання (для виходу з вікна відео натисніть q)'
          '\nДля виходу з програми напишіть exit.')
    while True:
        number = input('\n', )
        if number == '1':
            learn.learning_model()
            break

        elif number == '2':
            cluster.cluster_kmeans()
            break

        elif number == '3':
            download.download_video()
            break

        elif number == '4':
            recognizer.video_recognizer()
            break

        elif number == 'exit':
            break

        else:
            print('Невірний вибір, спробуйте ще.')

if __name__ == '__main__':

    choose_main()
