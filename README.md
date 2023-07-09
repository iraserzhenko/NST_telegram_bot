# NST_telegram_bot
Telegram bot on Python using Aiogram and PyTorch with Neural Stryle Transfer algorithm

Этот телеграм-бот умеет переносить стиль с одной фотографии на другую, используя Neural Style Transfer.

### Примеры работы:

Начальное фото:

<img src="https://github.com/iraserzhenko/NST_telegram_bot/blob/main/images/content_img1.jpg" width="500">

Фото стиля:

<img src="https://github.com/iraserzhenko/NST_telegram_bot/blob/main/images/style_img1.jpg" width="500">

Конечный результат:

<img src="https://github.com/iraserzhenko/NST_telegram_bot/blob/main/images/output_1.jpg" width="500">

### Запуск проекта:
---
1) Установите библиотеки pytorch, torchvision, aiogram
2) Скачайте проект, склонировав репозиторий
3) Перейдите в папку приложения в консоли
4) В файле `bot.py` задайте Ваш токен бота (API_TOKEN)
5) Запустите файл `main.py`
6) Для того, чтобы запустить бота, напишите ему команду `/style_transfer`
