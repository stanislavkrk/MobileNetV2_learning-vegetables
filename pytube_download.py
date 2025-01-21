import os

def download_video():
    # URL відео
    url_text = input("Введіть url відео з YouTube для завантаження: ")
    video_url = "https://www.youtube.com/watch?v=OxLWJ3iZLyg"  # Брокколі
    # video_url = "https://www.youtube.com/watch?v=RW4XMYz4cUA"    #перець
    # video_url = "https://www.youtube.com/watch?v=0uiO04BunRU"    #картопля
    # video_url = "https://www.youtube.com/watch?v=2vIUdxCfX9c"    #томат
    # video_url = "https://www.youtube.com/watch?v=2VKD8fAz-NE"    #гарбуз
    # video_url = url_text #Власний варіант

    # Завантаження відео
    os.system(f'yt-dlp -f best "{video_url}" -o "vegetables_video.mp4"')

    print("Відео успішно завантажено")

if __name__ == '__main__':

    download_video()