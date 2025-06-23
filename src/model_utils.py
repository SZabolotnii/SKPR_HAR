# -*- coding: utf-8 -*-
"""
Утиліти для збереження та завантаження моделей
"""

import pickle
import json
import os
import numpy as np


def save_model_safe(model_dict, filepath):
    """
    Безпечне збереження моделі без проблем з lambda функціями.
    
    Parameters:
    -----------
    model_dict : dict
        Словник з моделями
    filepath : str
        Шлях для збереження
    """
    try:
        # Спробуємо звичайне збереження
        with open(filepath, 'wb') as f:
            pickle.dump(model_dict, f)
    except (AttributeError, pickle.PicklingError):
        # Якщо не вдалося, зберігаємо лише параметри
        print("Не вдалося зберегти повну модель. Зберігаємо лише основні параметри...")
        
        # Створюємо альтернативний файл з параметрами
        params_path = filepath.replace('.pkl', '_params.json')
        
        params = {
            'feature_extractor_params': {},
            'classifier_params': {},
            'accuracy': None
        }
        
        # Зберігаємо параметри feature_extractor
        if 'feature_extractor' in model_dict:
            fe = model_dict['feature_extractor']
            params['feature_extractor_params'] = {
                'class_name': fe.__class__.__name__,
                'lambda_reg': getattr(fe, 'lambda_reg', None),
                'epsilon': getattr(fe, 'epsilon', None),
                'n': getattr(fe, 'n', None),
                'alpha': getattr(fe, 'alpha', None),
                'n_classes': len(getattr(fe, 'classes_', [])),
                'basis_powers': getattr(fe, 'basis_powers_', [])
            }
        
        # Зберігаємо параметри класифікатора
        if 'classifier' in model_dict:
            clf = model_dict['classifier']
            params['classifier_params'] = {
                'kernel': getattr(clf, 'kernel', None),
                'C': getattr(clf, 'C', None),
                'gamma': getattr(clf, 'gamma', None)
            }
        
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, default=str)
        
        print(f"Параметри моделі збережено в: {params_path}")


def save_experiment_models(results_dir, models, experiment_name):
    """
    Зберігає моделі експерименту з обробкою помилок.
    
    Parameters:
    -----------
    results_dir : str
        Директорія для результатів
    models : dict
        Словник з моделями
    experiment_name : str
        Назва експерименту
    """
    os.makedirs(f"{results_dir}/models", exist_ok=True)
    
    model_path = f"{results_dir}/models/{experiment_name}_model.pkl"
    save_model_safe(models, model_path)
    
    # Також зберігаємо основну інформацію в JSON
    info_path = f"{results_dir}/models/{experiment_name}_info.json"
    
    info = {
        'experiment': experiment_name,
        'models_included': list(models.keys()),
        'timestamp': str(np.datetime64('now'))
    }
    
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)


def load_model_safe(filepath):
    """
    Безпечне завантаження моделі.
    
    Parameters:
    -----------
    filepath : str
        Шлях до файлу моделі
        
    Returns:
    --------
    model_dict : dict or None
        Завантажена модель або None
    """
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Не вдалося завантажити модель: {e}")
            
            # Спробуємо завантажити параметри
            params_path = filepath.replace('.pkl', '_params.json')
            if os.path.exists(params_path):
                print(f"Завантажуємо параметри з: {params_path}")
                with open(params_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
    
    return None