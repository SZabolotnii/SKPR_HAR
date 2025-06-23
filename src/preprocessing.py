# -*- coding: utf-8 -*-
"""
Модуль з класами для попередньої обробки даних
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class StandardScaler3D(BaseEstimator, TransformerMixin):
    """
    Стандартизатор для 3D даних часових рядів.
    Масштабує кожен канал незалежно по всій вибірці.
    """
    
    def __init__(self):
        self.scalers_ = None
        
    def fit(self, X, y=None):
        """
        Обчислює параметри для стандартизації.
        
        Parameters:
        -----------
        X : array, shape (n_samples, n_timepoints, n_channels)
            3D масив сигналів
            
        Returns:
        --------
        self
        """
        n_channels = X.shape[2]
        
        # Створюємо окремий scaler для кожного каналу
        self.scalers_ = [StandardScaler() for _ in range(n_channels)]
        
        # Навчаємо кожен scaler на відповідному каналі
        for i in range(n_channels):
            # Отримуємо всі дані для i-го каналу
            channel_data = X[:, :, i]
            # Fit scaler на цих даних
            self.scalers_[i].fit(channel_data)
            
        return self
    
    def transform(self, X):
        """
        Трансформує дані використовуючи обчислені параметри.
        
        Parameters:
        -----------
        X : array, shape (n_samples, n_timepoints, n_channels)
            
        Returns:
        --------
        X_transformed : array, shape (n_samples, n_timepoints, n_channels)
            Стандартизовані дані
        """
        if self.scalers_ is None:
            raise ValueError("Scaler не навчений. Спочатку викличте fit().")
            
        X_transformed = np.zeros_like(X, dtype=float)
        
        # Трансформуємо кожен канал
        for i in range(X.shape[2]):
            X_transformed[:, :, i] = self.scalers_[i].transform(X[:, :, i])
            
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """Навчає та трансформує за один крок"""
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X):
        """
        Повертає дані до оригінального масштабу.
        
        Parameters:
        -----------
        X : array, shape (n_samples, n_timepoints, n_channels)
            
        Returns:
        --------
        X_original : array, shape (n_samples, n_timepoints, n_channels)
        """
        if self.scalers_ is None:
            raise ValueError("Scaler не навчений.")
            
        X_original = np.zeros_like(X)
        
        for i in range(X.shape[2]):
            X_original[:, :, i] = self.scalers_[i].inverse_transform(X[:, :, i])
            
        return X_original


class WindowNormalizer(BaseEstimator, TransformerMixin):
    """
    Нормалізує кожне вікно (зразок) незалежно.
    Корисно для врахування варіацій між різними записами.
    """
    
    def __init__(self, norm_type='zscore'):
        """
        Parameters:
        -----------
        norm_type : str, 'zscore' або 'minmax'
            Тип нормалізації
        """
        self.norm_type = norm_type
        
    def fit(self, X, y=None):
        # Цей трансформер не потребує навчання
        return self
    
    def transform(self, X):
        """
        Нормалізує кожен зразок незалежно.
        
        Parameters:
        -----------
        X : array, shape (n_samples, n_timepoints, n_channels)
            
        Returns:
        --------
        X_normalized : array, shape (n_samples, n_timepoints, n_channels)
        """
        X_normalized = np.zeros_like(X, dtype=float)
        
        for i in range(X.shape[0]):
            sample = X[i]
            
            if self.norm_type == 'zscore':
                # Z-score нормалізація
                mean = np.mean(sample, axis=0)
                std = np.std(sample, axis=0)
                std[std == 0] = 1  # Уникаємо ділення на нуль
                X_normalized[i] = (sample - mean) / std
                
            elif self.norm_type == 'minmax':
                # Min-max нормалізація
                min_val = np.min(sample, axis=0)
                max_val = np.max(sample, axis=0)
                range_val = max_val - min_val
                range_val[range_val == 0] = 1
                X_normalized[i] = (sample - min_val) / range_val
                
            else:
                raise ValueError(f"Невідомий тип нормалізації: {self.norm_type}")
                
        return X_normalized


class SignalDenoiser(BaseEstimator, TransformerMixin):
    """
    Застосовує фільтрацію для зменшення шуму в сигналах.
    """
    
    def __init__(self, method='moving_average', window_size=5):
        """
        Parameters:
        -----------
        method : str
            Метод фільтрації ('moving_average', 'median')
        window_size : int
            Розмір вікна для фільтрації
        """
        self.method = method
        self.window_size = window_size
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        Застосовує фільтрацію до сигналів.
        
        Parameters:
        -----------
        X : array, shape (n_samples, n_timepoints, n_channels)
            
        Returns:
        --------
        X_denoised : array, shape (n_samples, n_timepoints, n_channels)
        """
        from scipy.ndimage import uniform_filter1d, median_filter
        
        X_denoised = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                signal = X[i, :, j]
                
                if self.method == 'moving_average':
                    # Ковзне середнє
                    X_denoised[i, :, j] = uniform_filter1d(
                        signal, size=self.window_size, mode='nearest'
                    )
                elif self.method == 'median':
                    # Медіанний фільтр
                    X_denoised[i, :, j] = median_filter(
                        signal, size=self.window_size, mode='nearest'
                    )
                else:
                    raise ValueError(f"Невідомий метод: {self.method}")
                    
        return X_denoised


if __name__ == "__main__":
    # Тестування модуля
    print("Тестування модуля препроцесингу...")
    
    # Створюємо тестові дані
    np.random.seed(42)
    X_test = np.random.randn(100, 128, 9)
    
    # Тест StandardScaler3D
    print("\n1. Тестування StandardScaler3D:")
    scaler = StandardScaler3D()
    X_scaled = scaler.fit_transform(X_test)
    
    print(f"   Оригінальні дані - mean: {np.mean(X_test):.3f}, std: {np.std(X_test):.3f}")
    print(f"   Масштабовані дані - mean: {np.mean(X_scaled):.3f}, std: {np.std(X_scaled):.3f}")
    
    # Перевіряємо кожен канал
    print("\n   Перевірка по каналах:")
    for i in range(9):
        channel_mean = np.mean(X_scaled[:, :, i])
        channel_std = np.std(X_scaled[:, :, i])
        print(f"   Канал {i}: mean={channel_mean:.6f}, std={channel_std:.3f}")
    
    # Тест WindowNormalizer
    print("\n2. Тестування WindowNormalizer:")
    normalizer = WindowNormalizer(norm_type='zscore')
    X_normalized = normalizer.transform(X_test[:5])  # Тестуємо на 5 зразках
    
    print("   Перевірка нормалізації по зразках:")
    for i in range(5):
        sample_mean = np.mean(X_normalized[i])
        sample_std = np.std(X_normalized[i])
        print(f"   Зразок {i}: mean={sample_mean:.6f}, std={sample_std:.3f}")