# -*- coding: utf-8 -*-
"""
Експеримент 3: Оптимізація параметрів базису
Пошук оптимальних параметрів n та alpha для дробово-степеневого базису
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Додаємо шлях до модулів проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_raw_signals_and_labels, get_activity_names
from src.preprocessing import StandardScaler3D
from src.feature_extractors import BaseKunchenkoExtractor, AggregatedFeatureExtractor
from src.utils import plot_confusion_matrix, save_experiment_summary


class ParameterizedFeatureExtractor(AggregatedFeatureExtractor):
    """
    Версія екстрактора з параметризованими n та alpha
    """
    def __init__(self, n=3, alpha=0.0, lambda_reg=0.01, epsilon=1e-8):
        super().__init__(lambda_reg=lambda_reg, epsilon=epsilon)
        self.n = n
        self.alpha = alpha


def evaluate_parameters(n, alpha, X_train, y_train, X_val, y_val):
    """
    Оцінює продуктивність для заданих параметрів n та alpha.
    
    Returns:
    --------
    accuracy : float
        Точність на валідаційній вибірці
    """
    try:
        # Створюємо екстрактор з заданими параметрами
        extractor = ParameterizedFeatureExtractor(n=n, alpha=alpha)
        
        # Навчаємо та генеруємо ознаки
        extractor.fit(X_train, y_train)
        X_train_features = extractor.transform(X_train)
        X_val_features = extractor.transform(X_val)
        
        # Стандартизація
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_features)
        X_val_scaled = scaler.transform(X_val_features)
        
        # Навчання SVM
        classifier = SVC(kernel='rbf', C=100, gamma='scale', random_state=42)
        classifier.fit(X_train_scaled, y_train)
        
        # Оцінка
        y_pred = classifier.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, y_pred)
        
        return accuracy
        
    except Exception as e:
        print(f"Помилка для n={n}, alpha={alpha}: {e}")
        return 0.0


def run(results_dir=None):
    """
    Запускає експеримент з оптимізації параметрів базису.
    
    Parameters:
    -----------
    results_dir : str
        Директорія для збереження результатів
        
    Returns:
    --------
    results : dict
        Словник з результатами експерименту
    """
    
    print("Запуск експерименту з оптимізації параметрів базису...")
    
    # 1. Завантаження даних
    print("\n1. Завантаження даних...")
    X_train_full, y_train_full = load_raw_signals_and_labels("train")
    X_test, y_test = load_raw_signals_and_labels("test")
    
    # Розділяємо тренувальну вибірку на train/val для оптимізації
    n_train = int(0.8 * len(X_train_full))
    X_train = X_train_full[:n_train]
    y_train = y_train_full[:n_train]
    X_val = X_train_full[n_train:]
    y_val = y_train_full[n_train:]
    
    print(f"   Тренувальна: {X_train.shape}")
    print(f"   Валідаційна: {X_val.shape}")
    print(f"   Тестова: {X_test.shape}")
    
    # 2. Стандартизація
    print("\n2. Стандартизація 3D сигналів...")
    scaler_3d = StandardScaler3D()
    X_train_scaled = scaler_3d.fit_transform(X_train)
    X_val_scaled = scaler_3d.transform(X_val)
    X_test_scaled = scaler_3d.transform(X_test)
    
    # 3. Визначення сітки параметрів
    print("\n3. Визначення сітки параметрів для пошуку...")
    
    # Параметри для пошуку
    n_values = [2, 3, 4, 5]  # Кількість базисних функцій
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]  # Параметр alpha
    
    print(f"   n: {n_values}")
    print(f"   alpha: {alpha_values}")
    print(f"   Всього комбінацій: {len(n_values) * len(alpha_values)}")
    
    # 4. Grid search
    print("\n4. Виконання grid search...")
    
    results_grid = np.zeros((len(n_values), len(alpha_values)))
    
    best_accuracy = 0
    best_params = None
    
    for i, n in enumerate(n_values):
        for j, alpha in enumerate(alpha_values):
            print(f"\r   Тестування n={n}, alpha={alpha:.2f}...", end='')
            
            accuracy = evaluate_parameters(
                n, alpha, 
                X_train_scaled, y_train,
                X_val_scaled, y_val
            )
            
            results_grid[i, j] = accuracy
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'n': n, 'alpha': alpha}
    
    print(f"\n\nНайкращі параметри: n={best_params['n']}, alpha={best_params['alpha']:.2f}")
    print(f"Валідаційна точність: {best_accuracy:.4f}")
    
    # 5. Візуалізація результатів grid search
    print("\n5. Візуалізація результатів...")
    
    # Теплова карта
    plt.figure(figsize=(10, 8))
    sns.heatmap(results_grid, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=[f'{a:.2f}' for a in alpha_values],
                yticklabels=[f'n={n}' for n in n_values])
    plt.xlabel('Alpha', fontsize=12)
    plt.ylabel('Кількість базисних функцій (n)', fontsize=12)
    plt.title('Точність для різних комбінацій параметрів', fontsize=14)
    plt.tight_layout()
    
    if results_dir:
        plt.savefig(f"{results_dir}/figures/exp3_parameter_heatmap.png", 
                   dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3D візуалізація
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Створюємо сітку для поверхні
    N, A = np.meshgrid(n_values, alpha_values)
    Z = results_grid.T
    
    # Поверхня
    surf = ax.plot_surface(N, A, Z, cmap='viridis', alpha=0.8)
    
    # Позначаємо найкращу точку
    ax.scatter([best_params['n']], [best_params['alpha']], [best_accuracy],
               color='red', s=100, marker='*')
    
    ax.set_xlabel('n (кількість функцій)', fontsize=11)
    ax.set_ylabel('Alpha', fontsize=11)
    ax.set_zlabel('Точність', fontsize=11)
    ax.set_title('3D візуалізація простору параметрів', fontsize=14)
    
    # Додаємо колірну шкалу
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    if results_dir:
        plt.savefig(f"{results_dir}/figures/exp3_parameter_3d.png", 
                   dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Оцінка на тестовій вибірці з оптимальними параметрами
    print("\n6. Фінальна оцінка з оптимальними параметрами...")
    
    # Перенавчаємо на повній тренувальній вибірці
    X_train_full_scaled = scaler_3d.fit_transform(X_train_full)
    
    # Створюємо оптимальний екстрактор
    optimal_extractor = ParameterizedFeatureExtractor(
        n=best_params['n'], 
        alpha=best_params['alpha']
    )
    
    # Навчаємо та генеруємо ознаки
    optimal_extractor.fit(X_train_full_scaled, y_train_full)
    X_train_features = optimal_extractor.transform(X_train_full_scaled)
    X_test_features = optimal_extractor.transform(X_test_scaled)
    
    # Стандартизація та класифікація
    final_scaler = StandardScaler()
    X_train_final = final_scaler.fit_transform(X_train_features)
    X_test_final = final_scaler.transform(X_test_features)
    
    final_classifier = SVC(kernel='rbf', C=100, gamma='scale', random_state=42)
    final_classifier.fit(X_train_final, y_train_full)
    
    y_pred = final_classifier.predict(X_test_final)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== РЕЗУЛЬТАТ: Тестова точність = {test_accuracy:.4f} ===")
    
    # 7. Аналіз базисних функцій
    print("\n7. Аналіз оптимальних базисних функцій...")
    
    # Візуалізація базисних функцій
    x = np.linspace(-3, 3, 1000)
    
    plt.figure(figsize=(12, 5))
    
    # Графік функцій
    plt.subplot(1, 2, 1)
    for i in range(2, best_params['n'] + 1):
        p = optimal_extractor._compute_power(i, best_params['alpha'])
        y = np.sign(x) * (np.abs(x) + optimal_extractor.epsilon)**p
        plt.plot(x, y, label=f'φ_{i-1}(x), p={p:.3f}')
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('φ(x)', fontsize=12)
    plt.title(f'Оптимальні базисні функції (n={best_params["n"]}, α={best_params["alpha"]})',
              fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Графік степенів
    plt.subplot(1, 2, 2)
    n_range = range(2, 6)
    powers = [optimal_extractor._compute_power(i, best_params['alpha']) for i in n_range]
    plt.plot(n_range, powers, 'bo-', markersize=8)
    plt.xlabel('i', fontsize=12)
    plt.ylabel('Степінь p(i)', fontsize=12)
    plt.title(f'Залежність степеня від індексу (α={best_params["alpha"]})', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if results_dir:
        plt.savefig(f"{results_dir}/figures/exp3_optimal_basis_functions.png", 
                   dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. Матриця плутанини для фінальної моделі
    activity_map = get_activity_names()
    cm = plot_confusion_matrix(
        y_test, y_pred,
        class_names=list(activity_map.values()),
        title=f'Матриця плутанини - Оптимальні параметри\n'
              f'n={best_params["n"]}, α={best_params["alpha"]}, Точність={test_accuracy:.4f}',
        save_path=f"{results_dir}/figures/exp3_confusion_matrix.png" if results_dir else None
    )
    plt.show()
    
    # 9. Повертаємо результати
    results = {
        'best_params': best_params,
        'validation_accuracy': best_accuracy,
        'test_accuracy': test_accuracy,
        'accuracy': test_accuracy,  # для сумісності
        'parameter_grid': results_grid.tolist(),
        'n_values': n_values,
        'alpha_values': alpha_values,
        'confusion_matrix': cm.tolist()
    }
    
    if results_dir:
        save_experiment_summary(results, 
                              f"{results_dir}/reports/exp3_summary.txt")
    
    return results


if __name__ == "__main__":
    # Якщо запускається окремо
    results = run("results/exp3_optimal")
    print(f"\nОптимальні параметри: {results['best_params']}")
    print(f"Фінальна точність: {results['test_accuracy']:.4f}")