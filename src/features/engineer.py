import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Production-ready feature engineering class for the IEEE-CIS Fraud Detection dataset.
    Uses only pandas and numpy for maximum transparency and performance.
    """

    def __init__(self):
        """
        Initializes the FeatureEngineer with storage for learned parameters.
        """
        self.freq_maps = {}
        self.categorical_columns = ['ProductCD', 'card4']
        self.is_fitted = False

    def transform_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert TransactionDT (seconds) into hour_of_day and day_of_week.
        
        Args:
            df: Input DataFrame containing 'TransactionDT'.
            
        Returns:
            DataFrame with new time-based features.
        """
        if 'TransactionDT' in df.columns:
            # hour_of_day: 0-23
            df['hour_of_day'] = (df['TransactionDT'] // 3600) % 24
            # day_of_week: 0-6
            df['day_of_week'] = (df['TransactionDT'] // (3600 * 24)) % 7
        return df

    def scale_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize TransactionAmt using log1p transformation to handle outliers.
        
        Args:
            df: Input DataFrame containing 'TransactionAmt'.
            
        Returns:
            DataFrame with scaled transaction amounts.
        """
        if 'TransactionAmt' in df.columns:
            df['TransactionAmt'] = np.log1p(df['TransactionAmt'])
        return df

    def calculate_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the number of transactions per card (card1) within a daily window.
        
        Args:
            df: Input DataFrame containing 'card1' and 'TransactionDT'.
            
        Returns:
            DataFrame with card1_count_daily feature.
        """
        if 'card1' in df.columns and 'TransactionDT' in df.columns:
            # Create a daily bin (86400 seconds in a day)
            df['day_bin'] = df['TransactionDT'] // 86400
            df['card1_count_daily'] = df.groupby(['card1', 'day_bin'])['TransactionDT'].transform('count')
            df.drop(columns=['day_bin'], inplace=True)
        return df

    def encode_categorical(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Encode categorical features using Frequency Encoding.
        
        Args:
            df: Input DataFrame.
            fit: If True, learns the frequency mappings from the current data.
            
        Returns:
            DataFrame with encoded categorical features.
        """
        for col in self.categorical_columns:
            if col in df.columns:
                if fit:
                    # Learn frequency mapping: category -> percentage of occurrence
                    self.freq_maps[col] = df[col].value_counts(normalize=True).to_dict()
                
                # Apply mapping, filling unseen categories with 0 (since they have 0 freq in train)
                df[f'{col}_freq'] = df[col].map(self.freq_maps.get(col, {})).fillna(0)
        
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles NaNs: -999 for numericals and 'missing' for categoricals.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with missing values handled.
        """
        # Fill numerical columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(-999)
        
        # Fill categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        df[cat_cols] = df[cat_cols].fillna('missing')
        
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        A complete pipeline method that fits and transforms a raw dataframe.
        
        Args:
            df: Raw input DataFrame.
            
        Returns:
            Engineered DataFrame.
        """
        # Work on a copy to avoid SettingWithCopy warnings
        processed_df = df.copy()
        
        # 1. Handle missing values first to ensure smooth transformations
        processed_df = self.handle_missing_values(processed_df)
        
        # 2. Time transformations
        processed_df = self.transform_time(processed_df)
        
        # 3. Amount scaling
        processed_df = self.scale_amounts(processed_df)
        
        # 4. Velocity calculations
        processed_df = self.calculate_velocity(processed_df)
        
        # 5. Categorical encoding (with fitting)
        processed_df = self.encode_categorical(processed_df, fit=True)
        
        self.is_fitted = True
        return processed_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies learned transformations to a new (e.g. test) dataset.
        
        Args:
            df: Raw input DataFrame.
            
        Returns:
            Engineered DataFrame.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted (fit_transform) before calling transform.")
            
        processed_df = df.copy()
        processed_df = self.handle_missing_values(processed_df)
        processed_df = self.transform_time(processed_df)
        processed_df = self.scale_amounts(processed_df)
        processed_df = self.calculate_velocity(processed_df)
        processed_df = self.encode_categorical(processed_df, fit=False)
        
        return processed_df
