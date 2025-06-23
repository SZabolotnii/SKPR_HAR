# -*- coding: utf-8 -*-
"""
Експеримент 4: Гібридна модель
Поєднує 6 ознак SKPR з 24 найкращими традиційними ознаками
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Додаємо шлях до модулів проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_raw_signals_and_labels, load_561_features, get_activity_names
from src.preprocessing import StandardScaler3D
from src.feature_extractors import OptimalFeatureExtractor
from src.utils import plot_confusion_matrix, save_classification_report, plot_feature_importance


def run(results_dir=None):
    """
    Запускає експеримент з гібридною моделлю.
    
    Parameters:
    -----------
    results_dir : str
        Директорія для збереження результатів
        
    Returns:
    --------
    results : dict
        Словник з результатами експерименту
    """
    
    print("Запуск експерименту з гібридною моделлю...")
    
    # 1. Завантаження всіх типів даних
    print("\n1. Завантаження даних...")
    
    # Сирі сигнали для SKPR
    X_train_raw, y_train = load_raw_signals_and_labels("train")
    X_test_raw, y_test = load_raw_signals_and_labels("test")
    
    # 561 традиційна ознака
    X_train_561, feature_names = load_561_features("train")
    X_test_561, _ = load_561_features("test")
    
    print(f"   Сирі сигнали: {X_train_raw.shape}")
    print(f"   Традиційні ознаки: {X_train_561.shape}")
    
    # 2. Ідентифікація найкращих традиційних ознак
    print("\n2. Пошук найкращих традиційних ознак за допомогою Random Forest...")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_561, y_train)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Вибираємо топ-24 ознаки
    N_TOP_FEATURES = 24
    top_indices = indices[:N_TOP_FEATURES]
    
    print(f"\nТоп-{N_TOP_FEATURES} найважливіших традиційних ознак:")
    for i in range(min(10, N_TOP_FEATURES)):  # Показуємо перші 10
        idx = indices[i]
        print(f"   {i+1}. {feature_names[idx]} (важливість: {importances[idx]:.4f})")
    
    # Візуалізація важливості ознак
    if results_dir:
        plt.figure(figsize=(10, 8))
        plot_feature_importance(
            feature_names, importances, top_n=30,
            title="Топ-30 найважливіших традиційних ознак",
            save_path=f"{results_dir}/figures/exp4_feature_importance.png"
        )
        plt.show()
    
    # 3. Генерація SKPR ознак
    print("\n3. Генерація 6 ознак SKPR...")
    
    # Стандартизація сирих сигналів
    scaler_3d = StandardScaler3D()
    X_train_scaled = scaler_3d.fit_transform(X_train_raw)
    X_test_scaled = scaler_3d.transform(X_test_raw)
    
    # Генерація SKPR ознак
    feature_extractor = OptimalFeatureExtractor()
    feature_extractor.fit(X_train_scaled, y_train)
    X_train_kunchenko = feature_extractor.transform(X_train_scaled)
    X_test_kunchenko = feature_extractor.transform(X_test_scaled)
    
    print(f"   Згенеровано SKPR ознак: {X_train_kunchenko.shape}")
    
    # 4. Створення гібридного набору ознак
    print("\n4. Створення гібридного набору ознак...")
    
    # Відбираємо топ традиційні ознаки
    X_train_top_traditional = X_train_561[:, top_indices]
    X_test_top_traditional = X_test_561[:, top_indices]
    
    # Об'єднуємо SKPR та традиційні ознаки
    X_train_hybrid = np.hstack([X_train_kunchenko, X_train_top_traditional])
    X_test_hybrid = np.hstack([X_test_kunchenko, X_test_top_traditional])
    
    print(f"   Фінальний набір ознак: {X_train_hybrid.shape[1]} "
          f"(6 SKPR + {N_TOP_FEATURES} традиційних)")
    
    # 5. Стандартизація гібридних ознак
    print("\n5. Стандартизація гібридного набору...")
    final_scaler = StandardScaler()
    X_train_final = final_scaler.fit_transform(X_train_hybrid)
    X_test_final = final_scaler.transform(X_test_hybrid)
    
    # 6. Оптимізація гіперпараметрів SVM
    print("\n6. Пошук оптимальних параметрів SVM...")
    
    param_grid = {
        'C': [50, 100, 200],
        'gamma': ['scale', 0.01, 0.1]
    }
    
    grid_search = GridSearchCV(
        SVC(kernel='rbf', random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_final, y_train)
    
    print(f"\nНайкращі параметри: {grid_search.best_params_}")
    print(f"Найкраща крос-валідаційна точність: {grid_search.best_score_:.4f}")
    
    # 7. Оцінка на тестовій вибірці
    print("\n7. Оцінка фінальної моделі...")
    
    classifier = grid_search.best_estimator_
    y_pred = classifier.predict(X_test_final)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== РЕЗУЛЬТАТ: Точність гібридної моделі = {accuracy:.4f} ===")
    
    # 8. Детальний аналіз
    activity_map = get_activity_names()
    
    print("\nДетальний звіт по класах:")
    report = classification_report(y_test, y_pred, 
                                 target_names=activity_map.values(),
                                 output_dict=True)
    print(classification_report(y_test, y_pred, 
                              target_names=activity_map.values()))
    
    # 9. Візуалізація результатів
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=activity_map.values(),
                yticklabels=activity_map.values())
    plt.title(f'Матриця плутанини - Гібридна модель\n'
              f'Точність: {accuracy:.4f} (6 SKPR + {N_TOP_FEATURES} традиційних ознак)')
    plt.ylabel('Справжній клас')
    plt.xlabel('Передбачений клас')
    plt.tight_layout()
    
    # Зберігаємо результати
    if results_dir:
        plt.savefig(f"{results_dir}/figures/exp4_confusion_matrix.png", 
                   dpi=300, bbox_inches='tight')
        save_classification_report(report, 
                                 f"{results_dir}/reports/exp4_classification_report.txt")
        
        # Зберігаємо список використаних традиційних ознак
        with open(f"{results_dir}/reports/exp4_selected_features.txt", 'w', encoding='utf-8') as f:
            f.write(f"Топ-{N_TOP_FEATURES} традиційних ознак, використаних у гібридній моделі:\n\n")
            for i, idx in enumerate(top_indices):
                f.write(f"{i+1}. {feature_names[idx]} (важливість: {importances[idx]:.4f})\n")
    
    plt.show()
    
    # 10. Аналіз внеску різних типів ознак
    print("\n10. Аналіз внеску різних типів ознак...")
    
    # Категоризація традиційних ознак
    time_domain_count = sum(1 for idx in top_indices if 't' in feature_names[idx][:1])
    freq_domain_count = sum(1 for idx in top_indices if 'f' in feature_names[idx][:1])
    
    gravity_count = sum(1 for idx in top_indices if 'Gravity' in feature_names[idx])
    angle_count = sum(1 for idx in top_indices if 'angle' in feature_names[idx])
    energy_count = sum(1 for idx in top_indices if 'energy' in feature_names[idx])
    
    print(f"\nРозподіл відібраних традиційних ознак:")
    print(f"   - Часова область: {time_domain_count}")
    print(f"   - Частотна область: {freq_domain_count}")
    print(f"   - Гравітаційні: {gravity_count}")
    print(f"   - Кутові: {angle_count}")
    print(f"   - Енергетичні: {energy_count}")
    
    # 11. Повертаємо результати
    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'n_skpr_features': 6,
        'n_traditional_features': N_TOP_FEATURES,
        'total_features': X_train_hybrid.shape[1],
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'selected_traditional_features': [feature_names[idx] for idx in top_indices],
        'feature_distribution': {
            'time_domain': time_domain_count,
            'freq_domain': freq_domain_count,
            'gravity': gravity_count,
            'angle': angle_count,
            'energy': energy_count
        }
    }
    
    # Закриваємо всі фігури matplotlib
    plt.close('all')
    
    return results


if __name__ == "__main__":
    # Якщо запускається окремо
    results = run("results/exp4_hybrid")
    print(f"\nФінальна точність гібридної моделі: {results['accuracy']:.4f}")