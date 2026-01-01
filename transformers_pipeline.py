from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class GroupMeanDifference(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, value_col, output_col=None):
        self.group_col = group_col
        self.value_col = value_col
        self.output_col = output_col or f'{value_col}_diff_from_group_mean'

    def fit(self, X, y=None):
        self.group_means_ = (
            X.groupby(self.group_col)[self.value_col]
            .mean()
            .to_dict()
        )
        self.fitted_ = True
        return self

    def transform(self, X):
        X_ = X.copy()
        X_[self.output_col] = (
            X_[self.value_col] - X_[self.group_col].map(self.group_means_)
        )
        # print('GroupMeanDifference class: ',X_.columns)
        return X_

class LogDensityVolumeCalculator(BaseEstimator, TransformerMixin):
    def __init__(self,
                 weight_col='Final Weights in Grams',
                 dim_cols=('Length', 'Width', 'Height'),
                 log_volume_col='log_volume',
                 log_density_col='log_density_proxy',
                 density_col='density_proxy',
                 log_weight_col='log_final_weight'):
        self.weight_col = weight_col
        self.dim_cols = dim_cols
        self.log_volume_col = log_volume_col
        self.log_density_col = log_density_col
        self.density_col = density_col
        self.log_weight_col = log_weight_col
        self.epsilon = 1e-6

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame with named columns.")

        X_ = X.copy()

        # Compute volume
        volume = (
            X_[self.dim_cols[0]] *
            X_[self.dim_cols[1]] *
            X_[self.dim_cols[2]]
        )

        # Compute density proxy
        density_proxy = X_[self.weight_col] / (volume + self.epsilon)

        # Compute logs
        X_[self.log_volume_col] = np.log1p(volume + self.epsilon)
        X_[self.log_density_col] = np.log1p(density_proxy + self.epsilon)
        X_[self.log_weight_col] = np.log1p(X_[self.weight_col] + self.epsilon)

        # Add raw density column
        X_[self.density_col] = density_proxy

        # print('LogDensityVolumeCalculator class: ',X_.columns)
        return X_


class PricePerGramCalculator(BaseEstimator, TransformerMixin):
    def __init__(self,
                 price_col='Price In Dollar',
                 weight_col='Final Weights in Grams',
                 output_col='price_per_gram'):
        self.price_col = price_col
        self.weight_col = weight_col
        self.output_col = output_col
        self.epsilon = 1e-6

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame with named columns.")

        X_ = X.copy()
        X_[self.output_col] = X_[self.price_col] / (X_[self.weight_col] + self.epsilon)

        # print('PricePerGramCalculator: ',X_.columns)
        return X_  # Return full DataFrame



class AspectRatioCalculator(BaseEstimator, TransformerMixin):
    def __init__(self,
                 length_col='Length',
                 width_col='Width',
                 height_col='Height',
                 epsilon=1e-6):
        self.length_col = length_col
        self.width_col = width_col
        self.height_col = height_col
        self.epsilon = epsilon

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame with named columns.")

        X_ = X.copy()

        X_['L_by_W'] = X_[self.length_col] / (X_[self.width_col] + self.epsilon)
        X_['L_by_H'] = X_[self.length_col] / (X_[self.height_col] + self.epsilon)
        X_['W_by_H'] = X_[self.width_col]  / (X_[self.height_col] + self.epsilon)

        # print('AspectRatioCalculator: ',X_.columns)

        return X_

class HierarchyAggregator(BaseEstimator, TransformerMixin):
    def __init__(self,
                 group_col='Hierarchy',
                 agg_config=None):
        self.group_col = group_col
        self.agg_config = agg_config or {
            'Hierarchy_Weight_Mean': ('Final Weights in Grams', 'mean'),
            'Hierarchy_Price_Std': ('Price In Dollar', 'std'),
            'Hierarchy_Count': ('Hierarchy', 'count')
        }

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        # Group and aggregate based on config
        self.agg_df_ = X.groupby(self.group_col).agg(**self.agg_config).reset_index()
        self.fitted_ = True
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        X_ = X.copy()
        X_ = X_.merge(self.agg_df_, on=self.group_col, how='left')

        # print('HierarchyAggregator: ',X_.columns)

        return X_  # <- Return full DataFrame instead of just new_cols


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        a =  X.drop(columns=self.columns_to_drop, errors='ignore')
        # print('ColumnDropper: ',a.columns)
        return a
