# -*- coding: utf-8 -*-
"""
Експеримент 5: Інкрементальна валідація
Додає 6 ознак SKPR до повного набору з 561 традиційної ознаки
"""

import os
import sys
import numpy as np

# Налаштування matplotlib backend
import matplotlib
if not hasattr(sys, 'ps1'):  # Якщо не в інтерактивному режимі
    matplotlib.use('Agg')
    
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import stats

# Додаємо шлях до модулів проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_raw_signals_and_labels, load_561_features, get_activity_names
from src.preprocessing import StandardScaler3D
from src.feature_extractors import OptimalFeatureExtractor
from src.utils import plot_confusion_matrix, save_classification_report, save_experiment_summary


def mcnemar_test(y_true, y_pred1, y_pred2):
    """
    Виконує тест МакНемара для порівняння двох класифікаторів.
    
    Parameters:
    -----------
    y_true : array
        Справжні мітки
    y_pred1 : array
        Передбачення першого класифікатора
    y_pred2 : array
        Передбачення другого класифікатора
        
    Returns:
    --------
    statistic : float
        Статистика тесту
    p_value : float
        p-значення
    """
    # Створюємо таблицю непогоджень
    correct1 = y_pred1 == y_true
    correct2 = y_pred2 == y_true
    
    # b: перший правильний, другий неправильний
    # c: перший неправильний, другий правильний
    b = np.sum(correct1 & ~correct2)
    c = np.sum(~correct1 & correct2)
    
    # Тест МакНемара
    if b + c > 0:
        statistic = (abs(b - c) - 1)**2 / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
    else:
        statistic = 0
        p_value = 1.0
    
    return statistic, p_value, b, c


def run(results_dir=None):
    """
    Запускає експеримент з інкрементальною валідацією.
    
    Parameters:
    -----------
    results_dir : str
        Директорія для збереження результатів
        
    Returns:
    --------
    results : dict
        Словник з результатами експерименту
    """
    
    print("Запуск експерименту з інкрементальною валідацією...")
    print("Мета: довести унікальну інформаційну цінність ознак SKPR")
    
    # 1. Завантаження всіх даних
    print("\n1. Завантаження даних...")
    
    # Сирі сигнали для SKPR
    X_train_raw, y_train = load_raw_signals_and_labels("train")
    X_test_raw, y_test = load_raw_signals_and_labels("test")
    
    # 561 традиційна ознака
    X_train_561, feature_names = load_561_features("train")
    X_test_561, _ = load_561_features("test")
    
    print(f"   Сирі сигнали: {X_train_raw.shape}")
    print(f"   Традиційні ознаки: {X_train_561.shape}")
    print(f"   Кількість зразків: train={len(y_train)}, test={len(y_test)}")
    
    # 2. Базовий експеримент (тільки 561 ознака)
    print("\n2. БАЗОВИЙ ЕКСПЕРИМЕНТ: Оцінка на 561 традиційній ознаці...")
    
    # Стандартизація
    scaler_561 = StandardScaler()
    X_train_561_scaled = scaler_561.fit_transform(X_train_561)
    X_test_561_scaled = scaler_561.transform(X_test_561)
    
    # Навчання SVM
    print("   Навчання SVM на 561 ознаці...")
    classifier_base = SVC(kernel='rbf', C=100, gamma='scale', random_state=42)
    classifier_base.fit(X_train_561_scaled, y_train)
    
    # Прогнозування
    y_pred_base = classifier_base.predict(X_test_561_scaled)
    accuracy_base = accuracy_score(y_test, y_pred_base)
    
    print(f"\n   БАЗОВА ТОЧНІСТЬ (561 ознака): {accuracy_base:.4f}")
    
    # Детальний звіт
    activity_map = get_activity_names()
    report_base = classification_report(y_test, y_pred_base, 
                                      target_names=activity_map.values(),
                                      output_dict=True)
    
    # 3. Генерація 6 ознак SKPR
    print("\n3. Генерація 6 ознак SKPR...")
    
    # Стандартизація сирих сигналів
    scaler_3d = StandardScaler3D()
    X_train_raw_scaled = scaler_3d.fit_transform(X_train_raw)
    X_test_raw_scaled = scaler_3d.transform(X_test_raw)
    
    # Генерація SKPR ознак
    print("   Навчання оптимального генератора ознак...")
    feature_extractor = OptimalFeatureExtractor()
    feature_extractor.fit(X_train_raw_scaled, y_train)
    
    print("   Генерація ознак...")
    X_train_kunchenko = feature_extractor.transform(X_train_raw_scaled)
    X_test_kunchenko = feature_extractor.transform(X_test_raw_scaled)
    
    print(f"   Згенеровано ознак: {X_train_kunchenko.shape}")
    
    # 4. Розширений експеримент (561 + 6 ознак)
    print("\n4. РОЗШИРЕНИЙ ЕКСПЕРИМЕНТ: Оцінка на 567 ознаках (561 + 6 SKPR)...")
    
    # Об'єднання ознак
    X_train_extended = np.hstack([X_train_561, X_train_kunchenko])
    X_test_extended = np.hstack([X_test_561, X_test_kunchenko])
    
    print(f"   Розмір розширеного набору: {X_train_extended.shape}")
    
    # Стандартизація
    scaler_567 = StandardScaler()
    X_train_extended_scaled = scaler_567.fit_transform(X_train_extended)
    X_test_extended_scaled = scaler_567.transform(X_test_extended)
    
    # Навчання SVM
    print("   Навчання SVM на 567 ознаках...")
    classifier_extended = SVC(kernel='rbf', C=100, gamma='scale', random_state=42)
    classifier_extended.fit(X_train_extended_scaled, y_train)
    
    # Прогнозування
    y_pred_extended = classifier_extended.predict(X_test_extended_scaled)
    accuracy_extended = accuracy_score(y_test, y_pred_extended)
    
    print(f"\n   РОЗШИРЕНА ТОЧНІСТЬ (567 ознак): {accuracy_extended:.4f}")
    
    # Детальний звіт
    report_extended = classification_report(y_test, y_pred_extended,
                                          target_names=activity_map.values(),
                                          output_dict=True)
    
    # 5. Статистичний аналіз покращення
    print("\n5. СТАТИСТИЧНИЙ АНАЛІЗ ПОКРАЩЕННЯ...")
    
    improvement = accuracy_extended - accuracy_base
    improvement_percent = (improvement / accuracy_base) * 100
    
    print(f"\n   Абсолютне покращення: +{improvement:.4f}")
    print(f"   Відносне покращення: +{improvement_percent:.2f}%")
    
    # Тест МакНемара
    statistic, p_value, b, c = mcnemar_test(y_test, y_pred_base, y_pred_extended)
    
    print(f"\n   Тест МакНемара:")
    print(f"   - Базова модель права, розширена помиляється: {b} випадків")
    print(f"   - Базова модель помиляється, розширена права: {c} випадків")
    print(f"   - Статистика χ²: {statistic:.4f}")
    print(f"   - p-значення: {p_value:.4f}")
    
    if p_value < 0.05:
        print("   - Висновок: Покращення є СТАТИСТИЧНО ЗНАЧУЩИМ (p < 0.05)")
    else:
        print("   - Висновок: Покращення не є статистично значущим")
    
    # 6. Аналіз покращень по класах
    print("\n6. Аналіз покращень по класах...")
    
    class_improvements = {}
    for class_name in activity_map.values():
        f1_base = report_base[class_name]['f1-score']
        f1_extended = report_extended[class_name]['f1-score']
        improvement = f1_extended - f1_base
        class_improvements[class_name] = {
            'f1_base': f1_base,
            'f1_extended': f1_extended,
            'improvement': improvement
        }
    
    print("\nПокращення F1-score по класах:")
    for class_name, metrics in class_improvements.items():
        print(f"   {class_name}: {metrics['f1_base']:.3f} → {metrics['f1_extended']:.3f} "
              f"({metrics['improvement']:+.3f})")
    
    # 7. Візуалізація результатів
    print("\n7. Візуалізація результатів...")
    
    # Порівняння матриць плутанини
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Базова модель
    cm_base = confusion_matrix(y_test, y_pred_base)
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues',
                xticklabels=activity_map.values(),
                yticklabels=activity_map.values(),
                ax=ax1)
    ax1.set_title(f'Базова модель (561 ознака)\nТочність: {accuracy_base:.4f}')
    ax1.set_ylabel('Справжній клас')
    ax1.set_xlabel('Передбачений клас')
    
    # Розширена модель
    cm_extended = confusion_matrix(y_test, y_pred_extended)
    sns.heatmap(cm_extended, annot=True, fmt='d', cmap='Greens',
                xticklabels=activity_map.values(),
                yticklabels=activity_map.values(),
                ax=ax2)
    ax2.set_title(f'Розширена модель (567 ознак)\nТочність: {accuracy_extended:.4f}')
    ax2.set_ylabel('Справжній клас')
    ax2.set_xlabel('Передбачений клас')
    
    plt.tight_layout()
    
    if results_dir:
        plt.savefig(f"{results_dir}/figures/exp5_comparison.png", 
                   dpi=300, bbox_inches='tight')
    plt.show()
    
    # Різниця матриць
    plt.figure(figsize=(8, 6))
    cm_diff = cm_extended - cm_base
    sns.heatmap(cm_diff, annot=True, fmt='d', cmap='RdBu_r',
                xticklabels=activity_map.values(),
                yticklabels=activity_map.values(),
                center=0, vmin=-10, vmax=10)
    plt.title('Різниця матриць плутанини (Розширена - Базова)')
    plt.tight_layout()
    
    if results_dir:
        plt.savefig(f"{results_dir}/figures/exp5_matrix_difference.png", 
                   dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. Аналіз конкретних покращень
    print("\n8. Аналіз конкретних випадків покращення...")
    
    # Знаходимо зразки, де розширена модель виправила помилки
    corrected_samples = np.where((y_pred_base != y_test) & (y_pred_extended == y_test))[0]
    
    print(f"\nКількість виправлених помилок: {len(corrected_samples)}")
    
    if len(corrected_samples) > 0:
        # Аналізуємо типи виправлених помилок
        correction_matrix = np.zeros((6, 6))
        for idx in corrected_samples:
            true_class = y_test[idx]
            base_pred = y_pred_base[idx]
            correction_matrix[true_class, base_pred] += 1
        
        print("\nТипи виправлених помилок (рядки - справжні класи, стовпці - помилкові передбачення):")
        for i in range(6):
            for j in range(6):
                if correction_matrix[i, j] > 0:
                    print(f"   {activity_map[i]} ← {activity_map[j]}: "
                          f"{int(correction_matrix[i, j])} виправлень")
    
    # 9. Фінальні висновки
    print("\n" + "="*70)
    print("ФІНАЛЬНІ ВИСНОВКИ")
    print("="*70)
    
    print(f"\n1. Базова модель (561 ознака): {accuracy_base:.4f}")
    print(f"2. Розширена модель (567 ознак): {accuracy_extended:.4f}")
    print(f"3. Покращення: +{improvement:.4f} ({improvement_percent:.2f}%)")
    
    if improvement > 0:
        print("\n✓ ДОВЕДЕНО: 6 ознак SKPR містять УНІКАЛЬНУ інформацію,")
        print("  яка НЕ ПРИСУТНЯ в традиційному наборі з 561 ознаки!")
        if p_value < 0.05:
            print("  Покращення є статистично значущим (p < 0.05)")
    else:
        print("\n✗ Ознаки SKPR не покращили результат")
    
    # 10. Збереження результатів
    results = {
        'baseline_accuracy': accuracy_base,
        'enhanced_accuracy': accuracy_extended,
        'improvement': improvement,
        'improvement_percent': improvement_percent,
        'mcnemar_test': {
            'statistic': statistic,
            'p_value': p_value,
            'b': int(b),
            'c': int(c)
        },
        'class_improvements': class_improvements,
        'n_corrected_samples': len(corrected_samples)
    }
    
    if results_dir:
        save_experiment_summary(results, 
                              f"{results_dir}/reports/exp5_summary.txt")
        save_classification_report(report_base,
                                 f"{results_dir}/reports/exp5_baseline_report.txt")
        save_classification_report(report_extended,
                                 f"{results_dir}/reports/exp5_enhanced_report.txt")
    
    # Закриваємо всі фігури matplotlib
    plt.close('all')
    
    return results


if __name__ == "__main__":
    # Якщо запускається окремо
    results = run("results/exp5_validation")
    print(f"\nПокращення точності: +{results['improvement']:.4f}")