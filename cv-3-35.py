import cv2
import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
import sys
import math
import matplotlib.pyplot as plt

def initialize_video_writer(width, height, fps, output_file):
    if not all(isinstance(x, int) and x > 0 for x in [width, height, fps]):
        raise ValueError("width, height, fps должны быть положительными целыми числами")
    if not isinstance(output_file, str) or not output_file.strip():
        raise ValueError("output_file должен быть непустой строкой")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось инициализировать VideoWriter для {output_file}")
    return writer

def create_moving_shapes_config(spacing, height, kind='mixed'):
    """
    Возвращает список объектов (словарей) с начальными параметрами:
    kind: 'circles', 'rects', 'mixed'
    """
    if spacing <= 0 or height <= 0:
        raise ValueError("spacing и height > 0")
    speed = 5
    objs = []
    base_x = -spacing * 2
    for i, r in enumerate([20, 25, 15, 25, 30, 35, 15]):
        if kind == 'rects':
            # создаём прямоугольники (w,h)
            objs.append({'type':'rect', 'center':(base_x - spacing*i, height//2), 'size':(40 + (i%3)*10, 30 + (i%2)*10), 'speed':speed, 'color':(0,0,255)})
        elif kind == 'circles':
            objs.append({'type':'circle', 'center':(base_x - spacing*i, height//2), 'radius':r, 'speed':speed, 'color':(0,0,255)})
        else:
            if i % 2 == 0:
                objs.append({'type':'circle', 'center':(base_x - spacing*i, height//2 - 30*(i%2)), 'radius':r, 'speed':speed, 'color':(0,0,255)})
            else:
                objs.append({'type':'rect', 'center':(base_x - spacing*i, height//2 + 20*(i%2)), 'size':(40 + (i%3)*12, 30 + (i%2)*12), 'speed':speed, 'color':(0,0,255)})
    return objs

def draw_and_update_objects(frame, objs, width):
    if frame is None or not isinstance(frame, np.ndarray):
        raise ValueError("frame должен быть ndarray")
    for o in objs:
        x, y = o['center']
        x_new = x + o['speed']
        if x_new > width + 50:
            if o['type'] == 'circle':
                x_new = - (o['radius'] + 10)
            else:
                x_new = -60
        o['center'] = (x_new, y)
        if o['type'] == 'circle':
            cv2.circle(frame, (int(x_new), int(y)), int(o['radius']), o.get('draw_color', (0,0,0)), -1)
        else:
            w,h = o['size']
            top_left = (int(x_new - w//2), int(y - h//2))
            bottom_right = (int(x_new + w//2), int(y + h//2))
            cv2.rectangle(frame, top_left, bottom_right, o.get('draw_color', (0,0,0)), -1)

# Обнаружение и классификация
def preprocess_frame(frame, blur_ksize=5):
    """Возвращает серое, размытие для уменьшения шума."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur_ksize and blur_ksize%2==1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return gray

def get_binary_mask(gray, thresh=200):
    if gray is None:
        raise ValueError("gray is None")
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    # морфология для очистки
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def contour_area_filter(contours, min_area=200):
    return [c for c in contours if cv2.contourArea(c) >= min_area]

def classify_shape_by_contour(cnt, circularity_thresh=0.75):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri == 0 or area == 0:
        return 'other'
    circularity = (4 * math.pi * area) / (peri * peri)
    if circularity >= circularity_thresh:
        return 'circle'
    # approx polygon
    eps = 0.02 * peri
    approx = cv2.approxPolyDP(cnt, eps, True)
    if len(approx) == 4:
        return 'rectangle'
    return 'other'

# Цвет по HSV
DEFAULT_COLOR_RANGES = {
    'red': [ (np.array([0,50,50]), np.array([10,255,255])), (np.array([160,50,50]), np.array([180,255,255])) ],
    'green': (np.array([35,50,50]), np.array([85,255,255])),
    'blue': (np.array([100,50,50]), np.array([130,255,255])),
    'yellow': (np.array([18,50,50]), np.array([35,255,255]))
}

def detect_color_at_point(hsv_img, x, y, color_ranges=DEFAULT_COLOR_RANGES):
    h,w = hsv_img.shape[:2]
    if x<0 or y<0 or x>=w or y>=h:
        return 'unknown'
    px = hsv_img[int(y), int(x)]
    for name, rng in color_ranges.items():
        if isinstance(rng, list):
            for lower, upper in rng:
                if cv2.inRange(np.array([[px]]), lower, upper)[0,0] == 255:
                    return name
        else:
            lower, upper = rng
            if cv2.inRange(np.array([[px]]), lower, upper)[0,0] == 255:
                return name
    return 'unknown'

# Основная обработка видео
def process_video(input_video_path, output_annotated_path=None, stats_csv='stats.csv',
                  stats_json='stats.json', min_area=200, thresh=200, save_every_n_frames=1, show_window=False):

    if not isinstance(input_video_path, str) or not input_video_path.strip():
        raise ValueError("input_video_path должен быть непустой строкой")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео {input_video_path}")

    writer = None
    if output_annotated_path:
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = initialize_video_writer(width, height, fps, output_annotated_path)

    stats = []
    total_counter = {}

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # предобработка и маска
            gray = preprocess_frame(frame, blur_ksize=5)
            mask = get_binary_mask(gray, thresh=thresh)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contour_area_filter(contours, min_area=min_area)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            per_frame_counts = {'circle':0, 'rectangle':0, 'other':0}
            per_frame_color_shape = {}

            for cnt in contours:
                shape = classify_shape_by_contour(cnt, circularity_thresh=0.78)
                M = cv2.moments(cnt)
                if M.get('m00',0) != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                else:
                    x,y,w,h = cv2.boundingRect(cnt)
                    cx = x + w//2
                    cy = y + h//2

                color = detect_color_at_point(hsv, cx, cy)
                per_frame_counts[shape] = per_frame_counts.get(shape,0) + 1
                key = (shape, color)
                per_frame_color_shape[key] = per_frame_color_shape.get(key,0) + 1
                total_counter[key] = total_counter.get(key,0) + 1

                # аннотации на кадре
                cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
                label = f"{shape} {color}"
                cv2.putText(frame, label, (cx-40, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                cv2.circle(frame, (cx,cy), 3, (255,0,0), -1)

            # сохраняем статистику кадра
            stats.append({
                'frame': frame_idx,
                'total_objects': sum(per_frame_counts.values()),
                'circles': per_frame_counts.get('circle',0),
                'rectangles': per_frame_counts.get('rectangle',0),
                'others': per_frame_counts.get('other',0),
                'by_shape_color': per_frame_color_shape
            })

            # запишем аннотированный кадр каждое N-е
            if writer and (frame_idx % save_every_n_frames == 0):
                writer.write(frame)

            if show_window:
                cv2.imshow('Annotated', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()

    # собираем итоговую сводную статистику
    # преобразуем total_counter в удобный dict: shape -> total, and color breakdowns
    summary = {}
    for (shape,color), cnt in total_counter.items():
        summary.setdefault(shape, {'total':0, 'colors':{}} )
        summary[shape]['total'] += cnt
        summary[shape]['colors'][color] = summary[shape]['colors'].get(color,0) + cnt

    # Сохраняем stats в CSV/JSON
    try:
        df = pd.DataFrame([{
            'frame': s['frame'],
            'total_objects': s['total_objects'],
            'circles': s['circles'],
            'rectangles': s['rectangles'],
            'others': s['others']
        } for s in stats])
        df.to_csv(stats_csv, index=False)
    except Exception as e:
        print(f"Не удалось сохранить CSV: {e}", file=sys.stderr)

    try:
        with open(stats_json, 'w', encoding='utf-8') as f:
            json.dump({'frames': stats, 'summary': summary}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Не удалось сохранить JSON: {e}", file=sys.stderr)

    return summary, total_counter

# Визуализация статистики
def plot_and_save_summary(summary, total_counter, output_dir='output', prefix='summary'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1) Распределение по формам
    shapes = list(summary.keys())
    totals = [summary[s]['total'] for s in shapes]
    fig1, ax1 = plt.subplots(figsize=(6,4))
    ax1.bar(shapes, totals)
    ax1.set_title('Количество по формам')
    ax1.set_ylabel('Count')
    plt.tight_layout()
    f1 = Path(output_dir)/f"{prefix}_by_shape.png"
    fig1.savefig(f1, dpi=200)

    # 2) Распределение по цветам
    color_totals = {}
    for (shape,color), cnt in total_counter.items():
        color_totals[color] = color_totals.get(color,0) + cnt
    labels = list(color_totals.keys())
    sizes = [color_totals[k] for k in labels] if labels else []
    fig2, ax2 = plt.subplots(figsize=(6,6))
    if sizes:
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%')
        ax2.set_title('Распределение по цветам')
    else:
        ax2.text(0.5, 0.5, "No objects", ha='center')
    plt.tight_layout()
    f2 = Path(output_dir)/f"{prefix}_by_color.png"
    fig2.savefig(f2, dpi=200)

    plt.close(fig1)
    plt.close(fig2)
    return str(f1), str(f2)

# Печать итоговой строки
def pretty_print_summary(summary):
    circles = summary.get('circle',{}).get('total',0)
    rects = summary.get('rectangle',{}).get('total',0)
    others = summary.get('other',{}).get('total',0)
    print(f"Круги: {circles}, Прямоугольники: {rects}, Прочие: {others}")

def main_cli():
    parser = argparse.ArgumentParser(description='Conveyor inspection: counting + classification (shape & color)')
    parser.add_argument('--input', type=str, default=None, help='Путь к входному видео. Если не указан — генерируется синтетика.')
    parser.add_argument('--generate', action='store_true', help='Генерировать синтетическое видео (игнорирует --input).')
    parser.add_argument('--out-video', type=str, default='annotated_output.mp4', help='Путь для аннотированного видео (опционально).')
    parser.add_argument('--out-stats-csv', type=str, default='stats.csv', help='CSV файл статистики.')
    parser.add_argument('--out-stats-json', type=str, default='stats.json', help='JSON файл статистики.')
    parser.add_argument('--out-figures-dir', type=str, default='results', help='Папка для графиков.')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--duration', type=int, default=8, help='Длина синтетического видео в секундах (если generate).')
    parser.add_argument('--min-area', type=int, default=200)
    parser.add_argument('--thresh', type=int, default=200)
    parser.add_argument('--show', action='store_true', help='Показывать окно во время обработки')
    args = parser.parse_args()

    input_video = args.input
    generated_temp = None
    try:
        if args.generate or (not input_video):
            # сгенерируем синтетическое видео
            temp_path = 'synthetic_conveyor.mp4'
            generated_temp = temp_path
            num_frames = args.fps * args.duration
            writer = initialize_video_writer(args.width, args.height, args.fps, temp_path)
            objs = create_moving_shapes_config(spacing=120, height=args.height, kind='mixed')
            for f in range(num_frames):
                frame = np.ones((args.height, args.width, 3), dtype=np.uint8) * 255
                # для разнообразия назначим случайные цвета (red/green/blue/yellow) для фигур
                for i,o in enumerate(objs):
                    color_cycle = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]
                    o['draw_color'] = color_cycle[(i + (f//15)) % len(color_cycle)]
                draw_and_update_objects(frame, objs, args.width)
                writer.write(frame)
            writer.release()
            input_video = temp_path
            print(f"Синтетическое видео сохранено: {temp_path}")

        out_video = args.out_video if args.out_video else None
        summary, total_counter = process_video(input_video_path=input_video,
                                              output_annotated_path=out_video,
                                              stats_csv=args.out_stats_csv,
                                              stats_json=args.out_stats_json,
                                              min_area=args.min_area,
                                              thresh=args.thresh,
                                              show_window=args.show)
        pretty_print_summary(summary)
        f1, f2 = plot_and_save_summary(summary, total_counter, output_dir=args.out_figures_dir)
        print(f"Графики сохранены: {f1}, {f2}")
        print(f"CSV: {args.out_stats_csv}, JSON: {args.out_stats_json}")
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
    finally:
        pass

if __name__ == '__main__':
    main_cli()
