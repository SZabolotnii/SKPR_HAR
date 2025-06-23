#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для запуску експериментів в batch режимі без GUI
"""

import os
import sys

# Встановлюємо non-interactive backend для matplotlib
import matplotlib
matplotlib.use('Agg')

# Тепер можна імпортувати решту
from run_all_experiments import run_all_experiments

if __name__ == "__main__":
    print("Запуск експериментів в batch режимі (без GUI)...")
    print("Всі графіки будуть збережені в директорію results/")
    print("="*60)
    
    try:
        run_all_experiments()
        print("\n✓ Всі експерименти успішно завершено!")
    except Exception as e:
        print(f"\n✗ Помилка: {e}")
        import traceback
        traceback.print_exc()