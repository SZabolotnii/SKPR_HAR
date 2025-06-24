# -*- coding: utf-8 -*-
"""
Експеримент 2: Аналіз ансамблю експертів
Порівнює різні типи базисних функцій та їх ансамбль
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
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, classification_report

# Додаємо шлях до модулів проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_raw_signals_and_labels, get_activity_names
from src.preprocessing import StandardScaler3D
from src.feature_extractors import EnsembleExpert
from src.utils import plot_confusion_matrix, compare_models_barplot, save_experiment_summary


def run(results_dir=None):
    """
    Запускає експеимент з ансамблем експертів.
    
    Parameters:
    -----------
    results_dir : str
        Директорія для збереження результатів
        
    Returns:
    --------
    results : dict
        Словник з результатами експерименту
    """
    
    print("Запуск експерименту з ансамблем експертів...")
    
    if results_dir:
        os.makedirs(f"{results_dir}/figures", exist_ok=True)
        os.makedirs(f"{results_dir}/reports", exist_ok=True)
    
    # 1. Завантаження даних
    print("\n1. Завантаження даних...")
    X_train, y_train = load_raw_signals_and_labels("train")
    X_test, y_test = load_raw_signals_and_labels("test")
    
    # 2. Стандартизація
    print("\n2. Стандартизація 3D сигналів...")
    scaler_3d = StandardScaler3D()
    X_train_scaled = scaler_3d.fit_transform(X_train)
    X_test_scaled = scaler_3d.transform(X_test)
    
    # 3. Визначення експертів
    print("\n3. Визначення 4 типів експертів з різними базисними функціями...")
    
    experts = {
        "Polynomial": {
            "extractor": EnsembleExpert(basis_type='polynomial'),
            "description": "x², x³, x⁴"
        },
        "Trigonometric": {
            "extractor": EnsembleExpert(basis_type='trigonometric'),
            "description": "sin(x), sin(2x), sin(3x)"
        },
        "Robust": {
            "extractor": EnsembleExpert(basis_type='robust'),
            "description": "tanh(x), sigmoid(x), atanh(x)"
        },
        "Fractional": {
            "extractor": EnsembleExpert(basis_type='fractional'),
            "description": "√|x|, ∛|x|, ⁴√|x|"
        }
    }
    
    # 4. Оцінка кожного експерта окремо
    print("\n4. Оцінка індивідуальної продуктивності кожного експерта...")
    
    expert_results = {}
    all_predictions = {}
    
    for name, expert_info in experts.items():
        print(f"\n--- Експерт: {name} ---")
        print(f"   Базисні функції: {expert_info['description']}")
        
        # Навчання експерта
        expert = expert_info['extractor']
        expert.fit(X_train_scaled, y_train)
        
        # Генерація ознак
        X_train_expert = expert.transform(X_train_scaled)
        X_test_expert = expert.transform(X_test_scaled)
        
        # Стандартизація ознак
        scaler = StandardScaler()
        X_train_final = scaler.fit_transform(X_train_expert)
        X_test_final = scaler.transform(X_test_expert)
        
        # Навчання SVM
        classifier = SVC(kernel='rbf', C=100, gamma='scale', random_state=42)
        classifier.fit(X_train_final, y_train)
        
        # Прогнозування
        y_pred = classifier.predict(X_test_final)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   Точність: {accuracy:.4f}")
        
        expert_results[name] = accuracy
        all_predictions[name] = y_pred
        
        # Зберігаємо модель експерта
        expert_info['model'] = {
            'extractor': expert,
            'scaler': scaler,
            'classifier': classifier
        }
    
    # 5. Аналіз узгодженості експертів
    print("\n5. Аналіз узгодженості між експертами...")
    
    # Обчислюємо матрицю узгодженості
    agreement_matrix = np.zeros((len(experts), len(experts)))
    expert_names = list(experts.keys())
    
    for i, name1 in enumerate(expert_names):
        for j, name2 in enumerate(expert_names):
            agreement = np.mean(all_predictions[name1] == all_predictions[name2])
            agreement_matrix[i, j] = agreement
    
    # Візуалізація матриці узгодженості
    plt.figure(figsize=(8, 6))
    sns.heatmap(agreement_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=expert_names, yticklabels=expert_names,
                vmin=0.7, vmax=1.0)
    plt.title('Матриця узгодженості між експертами')
    plt.tight_layout()
    
    if results_dir:
        plt.savefig(f"{results_dir}/figures/exp2_expert_agreement.png", 
                   dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Створення та оцінка ансамблю
    print("\n6. Створення ансамблю всіх експертів...")
    
    # Створюємо FeatureUnion для об'єднання ознак від всіх експертів
    ensemble_features = FeatureUnion([
        ('poly', experts['Polynomial']['extractor']),
        ('trig', experts['Trigonometric']['extractor']),
        ('robust', experts['Robust']['extractor']),
        ('fractional', experts['Fractional']['extractor'])
    ])
    
    # Навчаємо ансамбль
    ensemble_features.fit(X_train_scaled, y_train)
    X_train_ensemble = ensemble_features.transform(X_train_scaled)
    X_test_ensemble = ensemble_features.transform(X_test_scaled)
    
    print(f"   Розмір об'єднаних ознак: {X_train_ensemble.shape}")
    
    # Стандартизація та класифікація
    ensemble_scaler = StandardScaler()
    X_train_ensemble_scaled = ensemble_scaler.fit_transform(X_train_ensemble)
    X_test_ensemble_scaled = ensemble_scaler.transform(X_test_ensemble)
    
    ensemble_classifier = SVC(kernel='rbf', C=100, gamma='scale', random_state=42)
    ensemble_classifier.fit(X_train_ensemble_scaled, y_train)
    
    y_pred_ensemble = ensemble_classifier.predict(X_test_ensemble_scaled)
    ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
    
    print(f"\n=== РЕЗУЛЬТАТ: Точність ансамблю = {ensemble_accuracy:.4f} ===")
    
    # 7. Порівняння результатів
    print("\n7. Порівняння результатів...")
    
    # Додаємо результат ансамблю
    all_results = expert_results.copy()
    all_results['Ensemble'] = ensemble_accuracy
    
    # Сортуємо за точністю
    sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
    
    print("\nРейтинг моделей за точністю:")
    for i, (name, acc) in enumerate(sorted_results):
        print(f"   {i+1}. {name}: {acc:.4f}")
    
    # Візуалізація порівняння
    model_names = list(all_results.keys())
    accuracies = list(all_results.values())
    
    plt.figure(figsize=(10, 6))
    compare_models_barplot(
        model_names, accuracies,
        title="Порівняння точності різних експертів та їх ансамблю",
        save_path=f"{results_dir}/figures/exp2_model_comparison.png" if results_dir else None
    )
    plt.show()
    
    # 8. Детальний аналіз найкращого експерта
    best_expert_name = sorted_results[0][0] if sorted_results[0][0] != 'Ensemble' else sorted_results[1][0]
    print(f"\n8. Детальний аналіз найкращого індивідуального експерта: {best_expert_name}")
    
    # Матриця плутанини для найкращого експерта
    activity_map = get_activity_names()
    cm = plot_confusion_matrix(
        y_test, all_predictions[best_expert_name],
        class_names=list(activity_map.values()),
        title=f'Матриця плутанини - Експерт {best_expert_name}',
        save_path=f"{results_dir}/figures/exp2_best_expert_cm.png" if results_dir else None
    )
    plt.show()
    
    # 9. Аналіз помилок ансамблю
    print("\n9. Аналіз покращень від ансамблю...")
    
    # Знаходимо випадки, де ансамбль виправив помилки індивідуальних експертів
    ensemble_correct = y_pred_ensemble == y_test
    
    improvements = {}
    for name, pred in all_predictions.items():
        expert_correct = pred == y_test
        improved = ensemble_correct & ~expert_correct
        improvements[name] = np.sum(improved)
    
    print("\nКількість помилок, виправлених ансамблем:")
    for name, count in improvements.items():
        print(f"   {name}: {count} випадків")
    
    # 10. Збереження результатів
    results = {
        'expert_results': expert_results,
        'ensemble_accuracy': ensemble_accuracy,
        'best_individual_expert': best_expert_name,
        'agreement_matrix': agreement_matrix.tolist(),
        'improvements_by_ensemble': improvements,
        'expert_descriptions': {name: info['description'] 
                              for name, info in experts.items()}
    }
    
    if results_dir:
        save_experiment_summary(results, 
                              f"{results_dir}/reports/exp2_summary.txt")
    
    # Закриваємо всі фігури matplotlib
    plt.close('all')
    
    return results


if __name__ == "__main__":
    # Якщо запускається окремо
    results = run("results/exp2_ensemble")
    print(f"\nФінальні результати:")
    print(f"Найкращий експерт: {results['best_individual_expert']} "
          f"({results['expert_results'][results['best_individual_expert']]:.4f})")
    print(f"Точність ансамблю: {results['ensemble_accuracy']:.4f}")