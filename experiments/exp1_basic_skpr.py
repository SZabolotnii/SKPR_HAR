# -*- coding: utf-8 -*-
"""
Експеримент 1: Базовий метод SKPR
Використовує 6 агрегованих ознак на основі похибки реконструкції
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Додаємо шлях до модулів проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_raw_signals_and_labels
from src.preprocessing import StandardScaler3D
from src.feature_extractors import AggregatedFeatureExtractor
from src.utils import plot_confusion_matrix, save_classification_report


def run(results_dir=None):
    """
    Запускає базовий експеримент SKPR.
    
    Parameters:
    -----------
    results_dir : str
        Директорія для збереження результатів
        
    Returns:
    --------
    results : dict
        Словник з результатами експерименту
    """
    
    print("Запуск базового експерименту SKPR...")
    
    # 1. Завантаження даних
    print("1. Завантаження даних...")
    X_train, y_train = load_raw_signals_and_labels("train")
    X_test, y_test = load_raw_signals_and_labels("test")
    print(f"   Розмір тренувальної вибірки: {X_train.shape}")
    print(f"   Розмір тестової вибірки: {X_test.shape}")
    
    # 2. Стандартизація сигналів
    print("\n2. Стандартизація 3D сигналів...")
    scaler_3d = StandardScaler3D()
    X_train_scaled = scaler_3d.fit_transform(X_train)
    X_test_scaled = scaler_3d.transform(X_test)
    
    # 3. Генерація ознак SKPR
    print("\n3. Генерація 6 агрегованих ознак SKPR...")
    feature_extractor = AggregatedFeatureExtractor(lambda_reg=0.01)
    
    print("   Навчання моделей реконструкції для кожного класу...")
    feature_extractor.fit(X_train_scaled, y_train)
    
    print("   Генерація ознак для тренувальної вибірки...")
    X_train_features = feature_extractor.transform(X_train_scaled)
    
    print("   Генерація ознак для тестової вибірки...")
    X_test_features = feature_extractor.transform(X_test_scaled)
    
    print(f"   Форма згенерованих ознак: {X_train_features.shape}")
    
    # 4. Стандартизація ознак
    print("\n4. Стандартизація згенерованих ознак...")
    scaler = StandardScaler()
    X_train_final = scaler.fit_transform(X_train_features)
    X_test_final = scaler.transform(X_test_features)
    
    # 5. Навчання SVM
    print("\n5. Навчання SVM класифікатора...")
    classifier = SVC(kernel='rbf', C=100, gamma='scale', random_state=42)
    classifier.fit(X_train_final, y_train)
    
    # 6. Прогнозування та оцінка
    print("\n6. Оцінка на тестовій вибірці...")
    y_pred = classifier.predict(X_test_final)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== РЕЗУЛЬТАТ: Точність = {accuracy:.4f} ===")
    
    # 7. Детальний аналіз
    activity_map = {
        0: 'WALKING',
        1: 'WALKING_UPSTAIRS', 
        2: 'WALKING_DOWNSTAIRS',
        3: 'SITTING',
        4: 'STANDING',
        5: 'LAYING'
    }
    
    print("\nДетальний звіт по класах:")
    report = classification_report(y_test, y_pred, 
                                 target_names=activity_map.values(),
                                 output_dict=True)
    print(classification_report(y_test, y_pred, 
                              target_names=activity_map.values()))
    
    # 8. Візуалізація матриці плутанини
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=activity_map.values(),
                yticklabels=activity_map.values())
    plt.title(f'Матриця плутанини - Базовий SKPR\nТочність: {accuracy:.4f}')
    plt.ylabel('Справжній клас')
    plt.xlabel('Передбачений клас')
    plt.tight_layout()
    
    # Зберігаємо результати
    if results_dir:
        plt.savefig(f"{results_dir}/figures/exp1_confusion_matrix.png", dpi=300, bbox_inches='tight')
        save_classification_report(report, f"{results_dir}/reports/exp1_classification_report.txt")
        
        # Зберігаємо модель
        import pickle
        with open(f"{results_dir}/models/exp1_model.pkl", 'wb') as f:
            pickle.dump({
                'feature_extractor': feature_extractor,
                'scaler': scaler,
                'classifier': classifier
            }, f)
    
    plt.show()
    
    # 9. Аналіз помилок
    print("\nАналіз основних помилок класифікації:")
    
    # Знаходимо найбільші помилки
    errors = []
    for i in range(6):
        for j in range(6):
            if i != j and cm[i, j] > 0:
                errors.append((cm[i, j], activity_map[i], activity_map[j]))
    
    errors.sort(reverse=True)
    
    print("\nТоп-5 найчастіших помилок:")
    for count, true_class, pred_class in errors[:5]:
        print(f"   {true_class} → {pred_class}: {count} випадків")
    
    # 10. Повертаємо результати
    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'feature_shape': X_train_features.shape,
        'model': {
            'feature_extractor': feature_extractor,
            'scaler': scaler,
            'classifier': classifier
        }
    }
    
    return results


if __name__ == "__main__":
    # Якщо запускається окремо
    results = run("results/exp1_basic")
    print(f"\nФінальна точність: {results['accuracy']:.4f}")