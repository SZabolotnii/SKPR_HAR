# -*- coding: utf-8 -*-
"""
Модуль для завантаження та підготовки даних UCI HAR Dataset
"""

import os
import io
import zipfile
import urllib.request
import numpy as np


# Константи
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "00240/UCI%20HAR%20Dataset.zip")
ROOT = "data/UCI_HAR_Dataset"
DATA_DIR = "data"


def download_har():
    """
    Завантажує датасет UCI HAR, якщо він ще не існує.
    """
    if os.path.isdir(ROOT):
        return
        
    print("⇩  Завантаження UCI HAR Dataset...")
    
    # Створюємо директорію для даних
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Завантажуємо архів
    with urllib.request.urlopen(URL) as r:
        with zipfile.ZipFile(io.BytesIO(r.read())) as z:
            z.extractall(DATA_DIR)
    
    # Перейменовуємо директорію
    old_path = os.path.join(DATA_DIR, "UCI HAR Dataset")
    if os.path.exists(old_path):
        os.rename(old_path, ROOT)
    
    print("✓  Датасет готовий до використання.")


def load_raw_signals_and_labels(split="train"):
    """
    Завантажує сирі сигнали сенсорів та мітки класів.
    
    Parameters:
    -----------
    split : str, 'train' або 'test'
        Вибір між тренувальною та тестовою вибірками
        
    Returns:
    --------
    X_3d : array, shape (n_samples, 128, 9)
        3D масив сигналів від 9 сенсорів
    y : array, shape (n_samples,)
        Мітки класів (0-5)
    """
    # Завантажуємо датасет, якщо потрібно
    download_har()
    
    # Завантажуємо мітки
    y = np.loadtxt(f"{ROOT}/{split}/y_{split}.txt").astype(int) - 1
    
    # Шлях до сигналів
    base_path = f"{ROOT}/{split}/Inertial Signals/"
    
    # Список файлів сигналів
    signal_files = [
        'body_acc_x', 'body_acc_y', 'body_acc_z',
        'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
        'total_acc_x', 'total_acc_y', 'total_acc_z'
    ]
    
    # Завантажуємо всі сигнали
    signal_data = []
    for name in signal_files:
        file_path = f"{base_path}{name}_{split}.txt"
        data = np.loadtxt(file_path)
        signal_data.append(data)
    
    # Об'єднуємо в 3D масив: (samples, time, channels)
    X_3d = np.stack(signal_data, axis=-1)
    
    return X_3d, y


def load_561_features(split="train"):
    """
    Завантажує попередньо обчислені 561 ознаку.
    
    Parameters:
    -----------
    split : str, 'train' або 'test'
        
    Returns:
    --------
    X : array, shape (n_samples, 561)
        Матриця ознак
    feature_names : array, shape (561,)
        Назви ознак
    """
    # Завантажуємо датасет, якщо потрібно
    download_har()
    
    # Завантажуємо матрицю ознак
    X = np.loadtxt(f"{ROOT}/{split}/X_{split}.txt")
    
    # Завантажуємо назви ознак
    feature_names = None
    feature_file = f"{ROOT}/features.txt"
    if os.path.exists(feature_file):
        feature_data = np.loadtxt(feature_file, dtype=str)
        feature_names = feature_data[:, 1]
    
    return X, feature_names


def get_activity_names():
    """
    Повертає словник з назвами активностей.
    
    Returns:
    --------
    activity_map : dict
        Маппінг від індексу до назви активності
    """
    activity_map = {
        0: 'WALKING',
        1: 'WALKING_UPSTAIRS',
        2: 'WALKING_DOWNSTAIRS',
        3: 'SITTING',
        4: 'STANDING',
        5: 'LAYING'
    }
    return activity_map


def load_subject_ids(split="train"):
    """
    Завантажує ідентифікатори суб'єктів для кожного зразка.
    
    Parameters:
    -----------
    split : str, 'train' або 'test'
        
    Returns:
    --------
    subject_ids : array, shape (n_samples,)
        ID суб'єктів (1-30)
    """
    # Завантажуємо датасет, якщо потрібно
    download_har()
    
    subject_ids = np.loadtxt(f"{ROOT}/{split}/subject_{split}.txt").astype(int)
    return subject_ids


def get_dataset_info():
    """
    Повертає інформацію про датасет.
    
    Returns:
    --------
    info : dict
        Словник з інформацією про датасет
    """
    # Завантажуємо дані для підрахунку
    X_train, y_train = load_raw_signals_and_labels("train")
    X_test, y_test = load_raw_signals_and_labels("test")
    
    info = {
        'n_train_samples': X_train.shape[0],
        'n_test_samples': X_test.shape[0],
        'n_timepoints': X_train.shape[1],
        'n_channels': X_train.shape[2],
        'n_classes': len(np.unique(y_train)),
        'sampling_rate': 50,  # Hz
        'window_size': 2.56,  # seconds
        'window_overlap': 0.5,  # 50%
        'channel_names': [
            'body_acc_x', 'body_acc_y', 'body_acc_z',
            'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
            'total_acc_x', 'total_acc_y', 'total_acc_z'
        ],
        'activity_names': get_activity_names()
    }
    
    return info


if __name__ == "__main__":
    # Тестування модуля
    print("Тестування модуля завантаження даних...")
    
    # Завантажуємо дані
    X_train, y_train = load_raw_signals_and_labels("train")
    X_test, y_test = load_raw_signals_and_labels("test")
    
    # Виводимо інформацію
    info = get_dataset_info()
    print("\nІнформація про датасет:")
    for key, value in info.items():
        if key != 'activity_names' and key != 'channel_names':
            print(f"  {key}: {value}")
    
    print("\nКанали сенсорів:")
    for i, name in enumerate(info['channel_names']):
        print(f"  {i}: {name}")
    
    print("\nКласи активностей:")
    for idx, name in info['activity_names'].items():
        count_train = np.sum(y_train == idx)
        count_test = np.sum(y_test == idx)
        print(f"  {idx}: {name} (train: {count_train}, test: {count_test})")