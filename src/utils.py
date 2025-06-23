# -*- coding: utf-8 -*-
"""
Допоміжні функції для візуалізації та звітності
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
import json


def plot_confusion_matrix(y_true, y_pred, class_names=None, title=None, 
                         figsize=(10, 8), save_path=None):
    """
    Створює та відображає матрицю плутанини.
    
    Parameters:
    -----------
    y_true : array
        Справжні мітки
    y_pred : array
        Передбачені мітки
    class_names : list
        Назви класів
    title : str
        Заголовок графіка
    figsize : tuple
        Розмір фігури
    save_path : str
        Шлях для збереження (опціонально)
    """
    # Обчислюємо матрицю плутанини
    cm = confusion_matrix(y_true, y_pred)
    
    # Створюємо фігуру
    plt.figure(figsize=figsize)
    
    # Використовуємо seaborn для красивої візуалізації
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                linewidths=0.5,
                annot_kws={"size": 12})
    
    # Додаємо заголовок
    if title:
        plt.title(title, fontsize=16, pad=20)
    
    plt.ylabel('Справжній клас', fontsize=14)
    plt.xlabel('Передбачений клас', fontsize=14)
    
    # Покращуємо читабельність
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Зберігаємо якщо потрібно
    if save_path:
        # Створюємо директорію якщо потрібно
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return cm


def save_classification_report(report_dict, save_path):
    """
    Зберігає classification report у текстовий файл.
    
    Parameters:
    -----------
    report_dict : dict
        Словник з classification_report(output_dict=True)
    save_path : str
        Шлях для збереження
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("Детальний звіт класифікації\n")
        f.write("=" * 70 + "\n\n")
        
        # Записуємо метрики для кожного класу
        f.write(f"{'Клас':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n")
        f.write("-" * 60 + "\n")
        
        for class_name, metrics in report_dict.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                f.write(f"{class_name:<20} "
                       f"{metrics['precision']:>10.4f} "
                       f"{metrics['recall']:>10.4f} "
                       f"{metrics['f1-score']:>10.4f} "
                       f"{metrics['support']:>10.0f}\n")
        
        f.write("-" * 60 + "\n")
        
        # Записуємо загальні метрики
        if 'accuracy' in report_dict:
            f.write(f"\nЗагальна точність: {report_dict['accuracy']:.4f}\n")
        
        if 'macro avg' in report_dict:
            f.write(f"\nMacro avg:    "
                   f"precision={report_dict['macro avg']['precision']:.4f}, "
                   f"recall={report_dict['macro avg']['recall']:.4f}, "
                   f"f1-score={report_dict['macro avg']['f1-score']:.4f}\n")
        
        if 'weighted avg' in report_dict:
            f.write(f"Weighted avg: "
                   f"precision={report_dict['weighted avg']['precision']:.4f}, "
                   f"recall={report_dict['weighted avg']['recall']:.4f}, "
                   f"f1-score={report_dict['weighted avg']['f1-score']:.4f}\n")


def plot_feature_importance(feature_names, importances, top_n=20, 
                          title="Важливість ознак", save_path=None):
    """
    Візуалізує важливість ознак.
    
    Parameters:
    -----------
    feature_names : list
        Назви ознак
    importances : array
        Важливість кожної ознаки
    top_n : int
        Кількість найважливіших ознак для відображення
    title : str
        Заголовок
    save_path : str
        Шлях для збереження
    """
    # Сортуємо за важливістю
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    
    # Створюємо барплот
    plt.barh(range(top_n), importances[indices], color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Важливість', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    
    # Додаємо значення на барах
    for i, v in enumerate(importances[indices]):
        plt.text(v + 0.001, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_learning_curves(train_scores, val_scores, train_sizes=None,
                        title="Криві навчання", save_path=None):
    """
    Візуалізує криві навчання.
    
    Parameters:
    -----------
    train_scores : array
        Точність на тренувальній вибірці
    val_scores : array
        Точність на валідаційній вибірці
    train_sizes : array
        Розміри тренувальної вибірки
    title : str
        Заголовок
    save_path : str
        Шлях для збереження
    """
    plt.figure(figsize=(10, 6))
    
    if train_sizes is None:
        train_sizes = range(1, len(train_scores) + 1)
    
    # Обчислюємо середні та стандартні відхилення
    train_mean = np.mean(train_scores, axis=1) if train_scores.ndim > 1 else train_scores
    train_std = np.std(train_scores, axis=1) if train_scores.ndim > 1 else np.zeros_like(train_mean)
    
    val_mean = np.mean(val_scores, axis=1) if val_scores.ndim > 1 else val_scores
    val_std = np.std(val_scores, axis=1) if val_scores.ndim > 1 else np.zeros_like(val_mean)
    
    # Малюємо криві
    plt.plot(train_sizes, train_mean, 'o-', color='royalblue', 
             label='Тренувальна точність', linewidth=2)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.1, color='royalblue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='forestgreen',
             label='Валідаційна точність', linewidth=2)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.1, color='forestgreen')
    
    plt.xlabel('Розмір тренувальної вибірки', fontsize=12)
    plt.ylabel('Точність', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def save_experiment_summary(results, save_path):
    """
    Зберігає загальне резюме експерименту.
    
    Parameters:
    -----------
    results : dict
        Словник з результатами
    save_path : str
        Шлях для збереження
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Конвертуємо numpy типи в звичайні Python типи
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Конвертуємо всі numpy типи
    results_converted = convert_numpy_types(results)
    
    # Зберігаємо в JSON
    json_path = save_path.replace('.txt', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_converted, f, ensure_ascii=False, indent=2)
    
    # Зберігаємо читабельний текстовий звіт
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("Резюме експерименту\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in results_converted.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value:.4f}\n")
            elif isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    # Зберігаємо читабельний текстовий звіт
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("Резюме експерименту\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in results.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value:.4f}\n")
            elif isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")


def compare_models_barplot(model_names, accuracies, title="Порівняння моделей",
                          save_path=None):
    """
    Створює барплот для порівняння точності різних моделей.
    
    Parameters:
    -----------
    model_names : list
        Назви моделей
    accuracies : list
        Точності моделей
    title : str
        Заголовок
    save_path : str
        Шлях для збереження
    """
    plt.figure(figsize=(10, 6))
    
    # Створюємо барплот
    bars = plt.bar(range(len(model_names)), accuracies, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    # Додаємо значення на барах
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=11)
    
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.ylabel('Точність', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.ylim(0, 1.05)
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # Тестування функцій візуалізації
    print("Тестування модуля візуалізації...")
    
    # Тестові дані
    np.random.seed(42)
    y_true = np.random.randint(0, 6, 100)
    y_pred = y_true.copy()
    # Додаємо деякі помилки
    errors = np.random.choice(100, 15, replace=False)
    y_pred[errors] = np.random.randint(0, 6, 15)
    
    activity_names = ['WALKING', 'WALKING_UP', 'WALKING_DOWN', 
                     'SITTING', 'STANDING', 'LAYING']
    
    # Тест матриці плутанини
    cm = plot_confusion_matrix(y_true, y_pred, class_names=activity_names,
                              title="Тестова матриця плутанини")
    plt.show()
    
    print("Візуалізація працює коректно!")