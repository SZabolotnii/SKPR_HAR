#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Головний скрипт для запуску всіх експериментів HAR-SKPR
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime

# Додаємо шлях до модулів проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Імпортуємо експерименти
from experiments import (
    exp1_basic_skpr,
    exp2_ensemble_analysis,
    exp3_optimal_basis,
    exp4_hybrid_model,
    exp5_full_validation
)

def create_results_directory():
    """Створює структуру директорій для результатів"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/run_{timestamp}"
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/figures", exist_ok=True)
    os.makedirs(f"{results_dir}/models", exist_ok=True)
    os.makedirs(f"{results_dir}/reports", exist_ok=True)
    
    return results_dir

def run_all_experiments():
    """Запускає всі експерименти послідовно"""
    print("="*70)
    print("HAR-SKPR: ПОВНИЙ НАБІР ЕКСПЕРИМЕНТІВ")
    print("="*70)
    
    # Створюємо директорію для результатів
    results_dir = create_results_directory()
    print(f"\nРезультати будуть збережені в: {results_dir}")
    
    # Словник для збереження всіх результатів
    all_results = {}
    
    # Експеримент 1: Базовий метод SKPR
    print("\n" + "="*70)
    print("ЕКСПЕРИМЕНТ 1: Базовий метод SKPR (6 агрегованих ознак)")
    print("="*70)
    start_time = time.time()
    try:
        results_1 = exp1_basic_skpr.run(results_dir)
        all_results['experiment_1'] = {
            'name': 'Базовий метод SKPR',
            'features': 6,
            'accuracy': results_1['accuracy'],
            'time': time.time() - start_time,
            'status': 'success'
        }
    except Exception as e:
        print(f"\n✗ Помилка в експерименті 1: {e}")
        all_results['experiment_1'] = {
            'name': 'Базовий метод SKPR',
            'error': str(e),
            'time': time.time() - start_time,
            'status': 'failed'
        }
    
    # Експеримент 2: Ансамбль експертів
    print("\n" + "="*70)
    print("ЕКСПЕРИМЕНТ 2: Аналіз ансамблю експертів")
    print("="*70)
    start_time = time.time()
    try:
        results_2 = exp2_ensemble_analysis.run(results_dir)
        all_results['experiment_2'] = {
            'name': 'Ансамбль експертів',
            'experts': results_2['expert_results'],
            'ensemble_accuracy': results_2['ensemble_accuracy'],
            'time': time.time() - start_time,
            'status': 'success'
        }
    except Exception as e:
        print(f"\n✗ Помилка в експерименті 2: {e}")
        all_results['experiment_2'] = {
            'name': 'Ансамбль експертів',
            'error': str(e),
            'time': time.time() - start_time,
            'status': 'failed'
        }
    
    # Експеримент 3: Оптимальний базис
    print("\n" + "="*70)
    print("ЕКСПЕРИМЕНТ 3: Оптимізація параметрів базису")
    print("="*70)
    start_time = time.time()
    try:
        results_3 = exp3_optimal_basis.run(results_dir)
        all_results['experiment_3'] = {
            'name': 'Оптимальний базис',
            'optimal_params': results_3['best_params'],
            'accuracy': results_3['accuracy'],
            'time': time.time() - start_time,
            'status': 'success'
        }
    except Exception as e:
        print(f"\n✗ Помилка в експерименті 3: {e}")
        all_results['experiment_3'] = {
            'name': 'Оптимальний базис',
            'error': str(e),
            'time': time.time() - start_time,
            'status': 'failed'
        }
    
    # Експеримент 4: Гібридна модель
    print("\n" + "="*70)
    print("ЕКСПЕРИМЕНТ 4: Гібридна модель (SKPR + традиційні ознаки)")
    print("="*70)
    start_time = time.time()
    try:
        results_4 = exp4_hybrid_model.run(results_dir)
        all_results['experiment_4'] = {
            'name': 'Гібридна модель',
            'skpr_features': 6,
            'traditional_features': 24,
            'total_features': 30,
            'accuracy': results_4['accuracy'],
            'time': time.time() - start_time,
            'status': 'success'
        }
    except Exception as e:
        print(f"\n✗ Помилка в експерименті 4: {e}")
        all_results['experiment_4'] = {
            'name': 'Гібридна модель',
            'error': str(e),
            'time': time.time() - start_time,
            'status': 'failed'
        }
    
    # Експеримент 5: Інкрементальна валідація
    print("\n" + "="*70)
    print("ЕКСПЕРИМЕНТ 5: Валідація на повному наборі ознак")
    print("="*70)
    start_time = time.time()
    try:
        results_5 = exp5_full_validation.run(results_dir)
        all_results['experiment_5'] = {
            'name': 'Інкрементальна валідація',
            'baseline_accuracy': results_5['baseline_accuracy'],
            'enhanced_accuracy': results_5['enhanced_accuracy'],
            'improvement': results_5['improvement'],
            'time': time.time() - start_time,
            'status': 'success'
        }
    except Exception as e:
        print(f"\n✗ Помилка в експерименті 5: {e}")
        all_results['experiment_5'] = {
            'name': 'Інкрементальна валідація',
            'error': str(e),
            'time': time.time() - start_time,
            'status': 'failed'
        }
    
    # Зберігаємо загальний звіт
    with open(f"{results_dir}/summary_report.json", 'w', encoding='utf-8') as f:
        # Конвертуємо numpy типи
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_numpy(all_results), f, ensure_ascii=False, indent=2)
    
    # Виводимо фінальне резюме
    print("\n" + "="*70)
    print("ФІНАЛЬНЕ РЕЗЮМЕ ВСІХ ЕКСПЕРИМЕНТІВ")
    print("="*70)
    
    successful = sum(1 for exp in all_results.values() if exp.get('status') == 'success')
    failed = sum(1 for exp in all_results.values() if exp.get('status') == 'failed')
    
    print(f"\nВиконано експериментів: {successful} успішно, {failed} з помилками")
    
    if all_results['experiment_1'].get('status') == 'success':
        print("\n1. Базовий метод SKPR:")
        print(f"   - Точність: {all_results['experiment_1']['accuracy']:.4f}")
    
    if all_results['experiment_2'].get('status') == 'success':
        print("\n2. Ансамбль експертів:")
        for expert, acc in all_results['experiment_2']['experts'].items():
            print(f"   - {expert}: {acc:.4f}")
        print(f"   - Ансамбль: {all_results['experiment_2']['ensemble_accuracy']:.4f}")
    
    if all_results['experiment_3'].get('status') == 'success':
        print("\n3. Оптимальний базис:")
        print(f"   - Параметри: {all_results['experiment_3']['optimal_params']}")
        print(f"   - Точність: {all_results['experiment_3']['accuracy']:.4f}")
    
    if all_results['experiment_4'].get('status') == 'success':
        print("\n4. Гібридна модель:")
        print(f"   - Точність: {all_results['experiment_4']['accuracy']:.4f}")
    
    if all_results['experiment_5'].get('status') == 'success':
        print("\n5. Інкрементальна валідація:")
        print(f"   - Базова точність (561 ознака): {all_results['experiment_5']['baseline_accuracy']:.4f}")
        print(f"   - Покращена точність (567 ознак): {all_results['experiment_5']['enhanced_accuracy']:.4f}")
        print(f"   - Приріст: +{all_results['experiment_5']['improvement']:.4f}")
    
    print(f"\nВсі результати збережено в: {results_dir}")
    
    if failed > 0:
        print(f"\n⚠ Увага: {failed} експеримент(ів) завершилися з помилками.")
        print("Перегляньте файл summary_report.json для деталей.")
    
    print("="*70)

if __name__ == "__main__":
    run_all_experiments()