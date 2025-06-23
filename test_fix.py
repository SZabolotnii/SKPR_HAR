#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тест виправлень проблеми з серіалізацією
"""

import sys
import os
import shutil

# Встановлюємо non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Додаємо шлях до модулів проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments import exp1_basic_skpr

if __name__ == "__main__":
    print("Тестування виправленого експерименту 1...")
    
    # Створюємо тестову директорію
    test_dir = "test_results"
    
    # Видаляємо стару директорію якщо існує
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Створюємо нову
    os.makedirs(test_dir, exist_ok=True)
    
    # Запускаємо експеримент 1
    try:
        results = exp1_basic_skpr.run(test_dir)
        print(f"\n✓ Експеримент успішно завершено!")
        print(f"Точність: {results['accuracy']:.4f}")
        
        # Перевіряємо створені файли
        print("\nСтворені файли:")
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                print(f"  {os.path.join(root, file)}")
                
    except Exception as e:
        print(f"\n✗ Помилка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Очищуємо після тесту
        if os.path.exists(test_dir):
            print(f"\nВидаляємо тестову директорію {test_dir}")
            shutil.rmtree(test_dir)