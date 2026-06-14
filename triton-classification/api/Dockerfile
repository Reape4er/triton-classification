# Базовый образ Python 3.10 (slim — минимальный размер)
FROM python:3.10-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Обновляем pip до последней версии
RUN pip install --upgrade pip

# Копируем файл зависимостей
COPY requirements.txt .
COPY class_names.json .
# Устанавливаем зависимости
# --no-cache-dir — не кэшировать (уменьшает размер образа)
# --default-timeout=100 — увеличенный таймаут для медленных сетей
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Копируем код приложения
COPY main.py .

# Открываем порт 8080
EXPOSE 8080

# Команда запуска
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]