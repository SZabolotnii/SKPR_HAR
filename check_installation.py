#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для перевірки правильності встановлення та налаштування проекту HAR-SKPR
"""

import sys
import os

def check_python_version():
    """Перевіряє версію Python"""
    print("1. Перевірка версії Python...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"   ✗ Python {version.major}.{version.minor}.{version.micro} - потрібна версія 3.8+")
        return False
    return True

def check_dependencies():
    """Перевіряє наявність необхідних бібліотек"""
    print("\n2. Перевірка залежностей...")
    
    dependencies = {
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'pandas': 'Pandas',
        'scipy': 'SciPy'
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"   ✓ {name}")
        except ImportError:
            print(f"   ✗ {name} не встановлено")
            all_ok = False
    
    return all_ok

def check_project_structure():
    """Перевіряє структуру проекту"""
    print("\n3. Перевірка структури проекту...")
    
    required_dirs = ['src', 'experiments', 'notebooks']
    required_files = [
        'run_all_experiments.py',
        'src/__init__.py',
        'src/data_loader.py',
        'src/preprocessing.py',
        'src/feature_extractors.py',
        'src/utils.py',
        'src/model_utils.py',
        'experiments/__init__.py',
        'experiments/exp1_basic_skpr.py'
    ]
    
    all_ok = True
    
    # Перевірка директорій
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"   ✓ Директорія {dir_name}/")
        else:
            print(f"   ✗ Відсутня директорія {dir_name}/")
            all_ok = False
    
    # Перевірка файлів
    for file_name in required_files:
        if os.path.isfile(file_name):
            print(f"   ✓ Файл {file_name}")
        else:
            print(f"   ✗ Відсутній файл {file_name}")
            all_ok = False
    
    return all_ok

def check_data_download():
    """Перевіряє можливість завантаження даних"""
    print("\n4. Перевірка завантаження даних...")
    
    try:
        from src.data_loader import get_dataset_info
        print("   ✓ Модуль data_loader імпортується")
        
        # Перевіряємо наявність даних
        if os.path.exists("data/UCI_HAR_Dataset"):
            print("   ✓ Датасет вже завантажено")
        else:
            print("   ⚠ Датасет ще не завантажено (буде завантажено при першому запуску)")
            
    except Exception as e:
        print(f"   ✗ Помилка імпорту: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Тестує базову функціональність"""
    print("\n5. Тест базової функціональності...")
    
    try:
        # Імпортуємо основні модулі
        from src.preprocessing import StandardScaler3D
        from src.feature_extractors import OptimalFeatureExtractor
        
        # Створюємо тестові дані
        import numpy as np
        X_test = np.random.randn(10, 128, 9)
        y_test = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
        
        # Тестуємо препроцесинг
        scaler = StandardScaler3D()
        X_scaled = scaler.fit_transform(X_test)
        print("   ✓ StandardScaler3D працює")
        
        # Тестуємо екстрактор ознак
        extractor = OptimalFeatureExtractor()
        extractor.fit(X_scaled, y_test)
        features = extractor.transform(X_scaled)
        print(f"   ✓ OptimalFeatureExtractor працює (згенеровано {features.shape[1]} ознак)")
        
    except Exception as e:
        print(f"   ✗ Помилка: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Головна функція перевірки"""
    print("="*60)
    print("ПЕРЕВІРКА ВСТАНОВЛЕННЯ HAR-SKPR")
    print("="*60)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_project_structure(),
        check_data_download(),
        test_basic_functionality()
    ]
    
    print("\n" + "="*60)
    if all(checks):
        print("✓ ВСЕ ГОТОВО ДО РОБОТИ!")
        print("\nТепер ви можете запустити експерименти:")
        print("  python run_all_experiments.py")
        print("\nАбо окремий експеримент:")
        print("  python experiments/exp1_basic_skpr.py")
    else:
        print("✗ ВИЯВЛЕНО ПРОБЛЕМИ")
        print("\nБудь ласка, виправте проблеми вище перед запуском експериментів.")
        print("Детальніше див. TROUBLESHOOTING.md")
    print("="*60)

if __name__ == "__main__":
    main()