import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения

# Используем OpenCV
img_cv = cv2.imread("1.jpg")  # заменить на свой путь к изображению

# Проверка успешной загрузки
if img_cv is None:
    raise ValueError("Изображение не найдено. Убедитесь, что путь указан правильно!")

# OpenCV загружает в формате BGR, переводим в RGB для корректного отображения
img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

# Также загрузим через Pillow
img_pil = Image.open("1.jpg")


# Преобразование в оттенки серого
gray_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
gray_pil = img_pil.convert("L")


# Изменение размера до 256x256
resized_cv = cv2.resize(gray_cv, (256, 256))
resized_pil = gray_pil.resize((256, 256))


# Визуализация
plt.figure(figsize=(10, 6))

# Исходное (OpenCV)
plt.subplot(2, 2, 1)
plt.imshow(img_cv_rgb)
plt.title("Исходное (OpenCV)")
plt.axis("off")

# В оттенках серого (OpenCV)
plt.subplot(2, 2, 2)
plt.imshow(gray_cv, cmap="gray")
plt.title("Gray (OpenCV)")
plt.axis("off")

# Исходное (Pillow)
plt.subplot(2, 2, 3)
plt.imshow(img_pil)
plt.title("Исходное (Pillow)")
plt.axis("off")

# В оттенках серого (Pillow)
plt.subplot(2, 2, 4)
plt.imshow(resized_pil, cmap="gray")
plt.title("Gray + Resize (Pillow)")
plt.axis("off")

plt.tight_layout()
plt.show()

# === 5. Сохранение результата ===
cv2.imwrite("result_cv_gray_256.jpg", resized_cv)   # OpenCV
resized_pil.save("result_pil_gray_256.jpg")         # Pillow
