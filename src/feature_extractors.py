# -*- coding: utf-8 -*-
"""
Модуль з класами для генерації ознак на основі SKPR
(Статистичного розпізнавання образів у просторі Кунченка)
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BaseKunchenkoExtractor(BaseEstimator, TransformerMixin):
    """
    Базовий клас для екстракції ознак методом SKPR.
    Реалізує основну логіку навчання моделей реконструкції для кожного класу.
    """
    
    def __init__(self, lambda_reg=0.01, epsilon=1e-8):
        self.lambda_reg = lambda_reg  # Параметр регуляризації
        self.epsilon = epsilon        # Для числової стабільності
        self.n = 3                   # Кількість базисних функцій
        self.alpha = 0.0             # Параметр для обчислення степенів
        
    def _compute_power(self, i, alpha):
        """Обчислює степінь для i-ої базисної функції"""
        A = 1/i
        B = 4 - i - (3/i)
        C = 2*i - 4 + (2/i)
        return A + B*alpha + C*(alpha**2)
    
    def _create_basis_function(self, power):
        """Створює базисну функцію з заданою степенню"""
        def basis_func(x):
            return np.sign(x) * (np.abs(x) + self.epsilon)**power
        return basis_func
    
    def fit(self, X_3d, y):
        """
        Навчає моделі реконструкції для кожного класу.
        
        Parameters:
        -----------
        X_3d : array, shape (n_samples, n_timepoints, n_channels)
            3D масив сигналів сенсорів
        y : array, shape (n_samples,)
            Мітки класів
        """
        # Зберігаємо параметри базисних функцій замість самих функцій
        self.basis_powers_ = []
        for i in range(2, self.n + 1):
            p = self._compute_power(i, self.alpha)
            self.basis_powers_.append(p)
        
        self.n_basis_funcs = len(self.basis_powers_)
        
        # Унікальні класи
        self.classes_ = np.unique(y)
        n_signals = X_3d.shape[2]
        
        # Словник для зберігання моделей кожного класу
        self.models_ = {}
        
        for c in self.classes_:
            X_class = X_3d[y == c]
            class_model = {}
            
            for signal_idx in range(n_signals):
                # Отримуємо всі сигнали даного типу для класу c
                all_signals = X_class[:, :, signal_idx]
                all_basis = self._apply_basis(all_signals)
                
                # Обчислюємо математичні сподівання
                E_x = np.mean(all_signals)
                E_phi = np.mean(all_basis, axis=(0, 1))
                
                # Центруємо дані
                centered_signals = all_signals - E_x
                centered_basis = all_basis - E_phi
                
                # Формуємо матриці для системи рівнянь
                n_pts = centered_signals.size
                flat_basis = centered_basis.reshape(n_pts, self.n_basis_funcs)
                flat_signals = centered_signals.flatten()
                
                # Матриця кореляцій F та вектор B
                F = flat_basis.T @ flat_basis / n_pts
                b = flat_basis.T @ flat_signals / n_pts
                
                # Регуляризована система
                F_reg = F + self.lambda_reg * np.eye(self.n_basis_funcs)
                K = np.linalg.solve(F_reg, b)
                
                # Зберігаємо параметри моделі
                class_model[signal_idx] = {
                    'K': K,
                    'E_x': E_x,
                    'E_phi': E_phi
                }
                
            self.models_[c] = class_model
            
        return self
    
    def _apply_basis(self, signal_data):
        """Застосовує базисні функції до сигналу"""
        # Відтворюємо базисні функції з збережених параметрів
        basis_results = []
        for power in self.basis_powers_:
            result = np.sign(signal_data) * (np.abs(signal_data) + self.epsilon)**power
            basis_results.append(result)
        return np.stack(basis_results, axis=-1)


class AggregatedFeatureExtractor(BaseKunchenkoExtractor):
    """
    Генерує 6 агрегованих ознак - по одній для кожного класу.
    Кожна ознака є логарифмом сумарної похибки реконструкції по всіх каналах.
    """
    
    def transform(self, X_3d):
        """
        Трансформує вхідні дані в вектор з 6 ознак.
        
        Returns:
        --------
        features : array, shape (n_samples, 6)
        """
        n_samples = X_3d.shape[0]
        features = np.zeros((n_samples, len(self.classes_)))
        
        for i in range(n_samples):
            for c_idx, c in enumerate(self.classes_):
                total_error = 0
                
                # Сумуємо похибки по всіх каналах
                for signal_idx in range(X_3d.shape[2]):
                    signal_1d = X_3d[i, :, signal_idx]
                    model = self.models_[c][signal_idx]
                    K, E_x, E_phi = model['K'], model['E_x'], model['E_phi']
                    
                    # Реконструкція сигналу
                    basis_matrix = self._apply_basis(signal_1d)
                    reconstructed_signal = E_x + (basis_matrix - E_phi) @ K
                    
                    # Додаємо MSE до загальної похибки
                    total_error += np.mean((signal_1d - reconstructed_signal)**2)
                
                # Логарифм для стабільності
                features[i, c_idx] = np.log(total_error + 1e-9)
                
        return features


class GroupedFeatureExtractor(BaseKunchenkoExtractor):
    """
    Генерує 18 ознак - по 3 групи каналів для кожного класу.
    Групи: акселерометр тіла, гіроскоп, загальне прискорення.
    """
    
    def transform(self, X_3d):
        """
        Трансформує вхідні дані в вектор з 18 ознак.
        
        Returns:
        --------
        features : array, shape (n_samples, 18)
        """
        n_samples = X_3d.shape[0]
        features = np.zeros((n_samples, len(self.classes_) * 3))
        
        # Визначаємо групи каналів
        signal_groups = {
            'body_acc': [0, 1, 2],      # body_acc_x, y, z
            'body_gyro': [3, 4, 5],     # body_gyro_x, y, z  
            'total_acc': [6, 7, 8]      # total_acc_x, y, z
        }
        
        for i in range(n_samples):
            feature_idx = 0
            
            for c_idx, c in enumerate(self.classes_):
                for group_name, group_indices in signal_groups.items():
                    group_error = 0
                    
                    # Обчислюємо похибку для групи каналів
                    for signal_idx in group_indices:
                        signal_1d = X_3d[i, :, signal_idx]
                        model = self.models_[c][signal_idx]
                        K, E_x, E_phi = model['K'], model['E_x'], model['E_phi']
                        
                        basis_matrix = self._apply_basis(signal_1d)
                        reconstructed_signal = E_x + (basis_matrix - E_phi) @ K
                        group_error += np.mean((signal_1d - reconstructed_signal)**2)
                    
                    # Усереднюємо по групі та логарифмуємо
                    features[i, feature_idx] = np.log(group_error / len(group_indices) + 1e-9)
                    feature_idx += 1
                    
        return features


class FullFeatureExtractor(BaseKunchenkoExtractor):
    """
    Генерує 54 ознаки - окрема похибка для кожного каналу та класу.
    Найбільш деталізований варіант.
    """
    
    def transform(self, X_3d):
        """
        Трансформує вхідні дані в вектор з 54 ознак.
        
        Returns:
        --------
        features : array, shape (n_samples, 54)
        """
        n_samples = X_3d.shape[0]
        n_channels = X_3d.shape[2]
        features = np.zeros((n_samples, len(self.classes_) * n_channels))
        
        for i in range(n_samples):
            feature_idx = 0
            
            for c_idx, c in enumerate(self.classes_):
                for signal_idx in range(n_channels):
                    signal_1d = X_3d[i, :, signal_idx]
                    model = self.models_[c][signal_idx]
                    K, E_x, E_phi = model['K'], model['E_x'], model['E_phi']
                    
                    basis_matrix = self._apply_basis(signal_1d)
                    reconstructed_signal = E_x + (basis_matrix - E_phi) @ K
                    error = np.mean((signal_1d - reconstructed_signal)**2)
                    
                    features[i, feature_idx] = np.log(error + 1e-9)
                    feature_idx += 1
                    
        return features


class OptimalFeatureExtractor(AggregatedFeatureExtractor):
    """
    Оптимізована версія з найкращими знайденими параметрами.
    Використовує дробово-степеневий базис з n=3, alpha=0.0.
    """
    
    def __init__(self, lambda_reg=0.01, epsilon=1e-8):
        super().__init__(lambda_reg=lambda_reg, epsilon=epsilon)
        # Оптимальні параметри знайдені експериментально
        self.n = 3
        self.alpha = 0.0


class EnsembleExpert(BaseKunchenkoExtractor):
    """
    Базовий клас для експерта з певним типом базисних функцій.
    """
    
    def __init__(self, basis_type='polynomial', lambda_reg=0.01):
        super().__init__(lambda_reg=lambda_reg)
        self.basis_type = basis_type
        
    def fit(self, X_3d, y):
        """Перевизначаємо для створення специфічних базисних функцій"""
        # Зберігаємо тип базису та його параметри
        self.basis_params_ = []
        
        if self.basis_type == 'polynomial':
            self.basis_params_ = [('power', 2), ('power', 3), ('power', 4)]
        elif self.basis_type == 'trigonometric':
            self.basis_params_ = [('sin', 1), ('cos', 1), ('sin', 2)]
        elif self.basis_type == 'robust':
            self.basis_params_ = [('tanh', 1), ('sigmoid', 1)]
        elif self.basis_type == 'fractional':
            self.basis_params_ = [('sqrt', 1), ('cbrt', 1)]
        
        self.n_basis_funcs = len(self.basis_params_)
        
        # Унікальні класи
        self.classes_ = np.unique(y)
        n_signals = X_3d.shape[2]
        
        # Словник для зберігання моделей кожного класу
        self.models_ = {}
        
        for c in self.classes_:
            X_class = X_3d[y == c]
            class_model = {}
            
            for signal_idx in range(n_signals):
                # Отримуємо всі сигнали даного типу для класу c
                all_signals = X_class[:, :, signal_idx]
                all_basis = self._apply_basis_ensemble(all_signals)
                
                # Обчислюємо математичні сподівання
                E_x = np.mean(all_signals)
                E_phi = np.mean(all_basis, axis=(0, 1))
                
                # Центруємо дані
                centered_signals = all_signals - E_x
                centered_basis = all_basis - E_phi
                
                # Формуємо матриці для системи рівнянь
                n_pts = centered_signals.size
                flat_basis = centered_basis.reshape(n_pts, self.n_basis_funcs)
                flat_signals = centered_signals.flatten()
                
                # Матриця кореляцій F та вектор B
                F = flat_basis.T @ flat_basis / n_pts
                b = flat_basis.T @ flat_signals / n_pts
                
                # Регуляризована система
                F_reg = F + self.lambda_reg * np.eye(self.n_basis_funcs)
                K = np.linalg.solve(F_reg, b)
                
                # Зберігаємо параметри моделі
                class_model[signal_idx] = {
                    'K': K,
                    'E_x': E_x,
                    'E_phi': E_phi
                }
                
            self.models_[c] = class_model
            
        return self
    
    def _apply_basis_ensemble(self, signal_data):
        """Застосовує базисні функції відповідно до типу"""
        basis_results = []
        
        for func_type, param in self.basis_params_:
            if func_type == 'power':
                result = signal_data ** param
            elif func_type == 'sin':
                result = np.sin(param * signal_data)
            elif func_type == 'cos':
                result = np.cos(param * signal_data)
            elif func_type == 'tanh':
                result = np.tanh(signal_data)
            elif func_type == 'sigmoid':
                result = 1 / (1 + np.exp(-signal_data))
            elif func_type == 'sqrt':
                result = np.sign(signal_data) * np.sqrt(np.abs(signal_data))
            elif func_type == 'cbrt':
                result = np.sign(signal_data) * np.cbrt(np.abs(signal_data))
            else:
                raise ValueError(f"Unknown function type: {func_type}")
            
            basis_results.append(result)
        
        return np.stack(basis_results, axis=-1)
    
    def transform(self, X_3d):
        """Використовуємо агреговану трансформацію"""
        n_samples = X_3d.shape[0]
        features = np.zeros((n_samples, len(self.classes_)))
        
        for i in range(n_samples):
            for c_idx, c in enumerate(self.classes_):
                total_error = 0
                
                # Сумуємо похибки по всіх каналах
                for signal_idx in range(X_3d.shape[2]):
                    signal_1d = X_3d[i, :, signal_idx]
                    model = self.models_[c][signal_idx]
                    K, E_x, E_phi = model['K'], model['E_x'], model['E_phi']
                    
                    # Реконструкція сигналу
                    basis_matrix = self._apply_basis_ensemble(signal_1d)
                    reconstructed_signal = E_x + (basis_matrix - E_phi) @ K
                    
                    # Додаємо MSE до загальної похибки
                    total_error += np.mean((signal_1d - reconstructed_signal)**2)
                
                # Логарифм для стабільності
                features[i, c_idx] = np.log(total_error + 1e-9)
                
        return features