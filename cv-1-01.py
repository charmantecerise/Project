import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Загрузка изображения с помощью OpenCV
def load_image_cv(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {path}")
    return img

# Загрузка изображения с помощью Pillow
def load_image_pil(path: str):
    try:
        return Image.open(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден: {path}")

# Преобразование в серый через OpenCV
def convert_to_gray_cv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Преобразование в серый через Pillow
def convert_to_gray_pil(img):
    return img.convert("L")

# Изменение размера через OpenCV
def resize_image_cv(img, size=(256, 256)):
    return cv2.resize(img, size)

# Изменение размера через Pillow
def resize_image_pil(img, size=(256, 256)):
    return img.resize(size)

# Визуализация изображений
def visualize(img_cv_rgb, gray_cv, img_pil, resized_pil, size):
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(img_cv_rgb)
    plt.title("Исходное (OpenCV)")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(gray_cv, cmap="gray")
    plt.title("Gray (OpenCV)")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(img_pil)
    plt.title("Исходное (Pillow)")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(resized_pil, cmap="gray")
    plt.title(f"Gray + Resize {size} (Pillow)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Основной пайплайн: загрузка, преобразование, визуализация, сохранение
def process_image(input_path: str,
                  output_cv_path: str = "result_cv_gray.jpg",
                  output_pil_path: str = "result_pil_gray.jpg",
                  size=(256, 256),
                  show=True):

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    # OpenCV
    img_cv = load_image_cv(str(input_path))
    img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    gray_cv = convert_to_gray_cv(img_cv)
    resized_cv = resize_image_cv(gray_cv, size)
    cv2.imwrite(output_cv_path, resized_cv)

    # Pillow
    img_pil = load_image_pil(str(input_path))
    gray_pil = convert_to_gray_pil(img_pil)
    resized_pil = resize_image_pil(gray_pil, size)
    resized_pil.save(output_pil_path)

    if show:
        visualize(img_cv_rgb, gray_cv, img_pil, resized_pil, size)

    print(f"✅ Сохранено: {output_cv_path}, {output_pil_path}")


if __name__ == "__main__":
    process_image(
        input_path="1.jpg",
        output_cv_path="result_cv_gray_256.jpg",
        output_pil_path="result_pil_gray_256.jpg",
        size=(256, 256),
        show=True
    )
