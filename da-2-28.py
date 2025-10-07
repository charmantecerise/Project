import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

# Создает синтетический временной ряд
def create_synthetic_series(n_points, start_date='2024-01-01'):
    try:
        if n_points <= 0:
            raise ValueError("Количество точек должно быть положительным числом")

        np.random.seed(42)
        dates = pd.date_range(start=start_date, periods=n_points, freq='D')
        values = np.random.randint(10, 100, size=n_points)
        df = pd.DataFrame({'date': dates, 'value': values})
        return df
    except Exception as e:
        raise Exception(f"Ошибка при создании временного ряда: {e}")

# Добавляет признаки: скользящее среднее и количество значений > среднего
def calculate_features(df, window, value_col='value'):
    try:
        if value_col not in df.columns:
            raise KeyError(f"Столбец {value_col} не найден в DataFrame")

        df = df.copy()
        df['rolling_mean'] = df[value_col].rolling(window=window, min_periods=1).mean()

        def count_above_mean(x):
            mean_val = x.mean()
            return np.sum(x > mean_val)

        df['count_above_mean'] = (
            df[value_col].rolling(window=window, min_periods=1)
            .apply(count_above_mean, raw=False)
        )

        return df
    except Exception as e:
        raise Exception(f"Ошибка при расчёте признаков: {e}")

# Строит и сохраняет график временного ряда и новых признаков
def build_graph(df, output_dir, format_file, date_col='date', value_col='value'):
    try:
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df[date_col], df[value_col], label='Исходные данные', linewidth=2)
        plt.plot(df[date_col], df['rolling_mean'], label='Скользящее среднее', linewidth=2)
        plt.plot(df[date_col], df['count_above_mean'],
                 label='Кол-во > среднего', linestyle='--', linewidth=2)

        plt.title(f'Скользящее окно: количество значений > среднего')
        plt.xlabel('Дата')
        plt.ylabel('Значения')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        output_path = Path(output_dir) / f"rolling_count_above_mean.{format_file}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, format=format_file, dpi=300)
        plt.show()

        return fig
    except Exception as e:
        raise Exception(f"Ошибка при построении графика: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Создание признака: количество значений > среднего в скользящем окне'
    )
    parser.add_argument('-n', '--n-points', type=int, default=50,
                        help='Количество точек временного ряда (по умолчанию: 50)')
    parser.add_argument('-w', '--window', type=int, default=5,
                        help='Размер окна (по умолчанию: 5)')
    parser.add_argument('-s', '--start-date', default='2024-01-01',
                        help='Начальная дата временного ряда (по умолчанию: 2024-01-01)')
    parser.add_argument('-o', '--output', default='output',
                        help='Директория для сохранения графика (по умолчанию: output)')
    parser.add_argument('--format', default='png', choices=['png', 'pdf', 'jpg'],
                        help='Формат файла (png, pdf, jpg)')

    args = parser.parse_args()

    try:
        print("Создаём временной ряд...")
        df = create_synthetic_series(args.n_points, args.start_date)

        print("Вычисляем признаки...")
        df = calculate_features(df, args.window)

        print("Строим график...")
        build_graph(df, args.output, args.format)

        print(f"✅ Готово! График сохранён в {args.output}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
