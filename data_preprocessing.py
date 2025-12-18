"""
Data Preprocessing Module for Sepsis Prediction
Handles data cleaning, feature engineering, and sequence preparation
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class SepsisDataPreprocessor:
    """Preprocesses clinical data for sepsis prediction"""
    
    def __init__(self, prediction_horizon: int = 6):
        """
        Args:
            prediction_horizon: Hours ahead to predict sepsis (default: 6)
        """
        self.prediction_horizon = prediction_horizon
        self.feature_columns = None
        self.scaler_params = {}
        
    def load_patient_data(self, file_path: str) -> pd.DataFrame:
        """Load a single patient's data from .psv file"""
        try:
            df = pd.read_csv(file_path, sep='|')
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare patient data"""
        if df is None or df.empty:
            return None
            
        # Make a copy
        df_clean = df.copy()
        
        # Remove rows where all clinical features are NaN
        clinical_cols = [col for col in df_clean.columns 
                        if col not in ['Age', 'Gender', 'Unit1', 'Unit2', 
                                      'HospAdmTime', 'ICULOS', 'SepsisLabel']]
        df_clean = df_clean.dropna(subset=clinical_cols, how='all')
        
        # Forward fill missing values (carry forward last known value)
        df_clean[clinical_cols] = df_clean[clinical_cols].ffill()
        
        # Backward fill remaining NaN values
        df_clean[clinical_cols] = df_clean[clinical_cols].bfill()
        
        # Fill remaining NaN with column median
        for col in clinical_cols:
            if df_clean[col].isna().any():
                median_val = df_clean[col].median()
                if pd.notna(median_val):
                    df_clean[col].fillna(median_val, inplace=True)
                else:
                    df_clean[col].fillna(0, inplace=True)
        
        # Ensure numeric types
        for col in clinical_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col].fillna(0, inplace=True)
        
        # Handle categorical features
        if 'Gender' in df_clean.columns:
            df_clean['Gender'] = df_clean['Gender'].fillna(0).astype(int)
        if 'Unit1' in df_clean.columns:
            df_clean['Unit1'] = pd.to_numeric(df_clean['Unit1'], errors='coerce').fillna(0)
        if 'Unit2' in df_clean.columns:
            df_clean['Unit2'] = pd.to_numeric(df_clean['Unit2'], errors='coerce').fillna(0)
        
        return df_clean
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create prediction labels: 1 if sepsis occurs within next 6 hours, else 0
        """
        df_labeled = df.copy()
        
        if 'SepsisLabel' not in df_labeled.columns:
            df_labeled['SepsisLabel'] = 0
        
        # Create target: sepsis in next 6 hours
        df_labeled['Target'] = 0
        
        # For each time point, check if sepsis occurs in next 6 hours
        for idx in range(len(df_labeled)):
            future_window = df_labeled.iloc[idx:idx+self.prediction_horizon+1]
            if future_window['SepsisLabel'].sum() > 0:
                df_labeled.loc[df_labeled.index[idx], 'Target'] = 1
        
        return df_labeled
    
    def extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from cleaned data
        
        Returns:
            features: numpy array of shape (n_timesteps, n_features)
            labels: numpy array of shape (n_timesteps,)
        """
        if df is None or df.empty:
            return None, None
        
        # Define feature columns (exclude metadata and target)
        exclude_cols = ['SepsisLabel', 'Target', 'ICULOS', 'HospAdmTime']
        
        if self.feature_columns is None:
            self.feature_columns = [col for col in df.columns 
                                   if col not in exclude_cols]
        
        # Extract features
        features = df[self.feature_columns].values.astype(np.float32)
        
        # Extract labels
        if 'Target' in df.columns:
            labels = df['Target'].values.astype(np.int32)
        else:
            labels = np.zeros(len(df), dtype=np.int32)
        
        return features, labels
    
    def normalize_features(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize features using z-score normalization
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            fit: If True, fit scaler; if False, use existing scaler params
            
        Returns:
            Normalized features
        """
        if features is None or len(features) == 0:
            return features
        
        features_norm = features.copy()
        
        if fit:
            # Calculate mean and std for each feature
            self.scaler_params['mean'] = np.nanmean(features, axis=0)
            self.scaler_params['std'] = np.nanstd(features, axis=0)
            # Avoid division by zero
            self.scaler_params['std'] = np.where(
                self.scaler_params['std'] == 0, 
                1.0, 
                self.scaler_params['std']
            )
        
        # Normalize
        features_norm = (features_norm - self.scaler_params['mean']) / self.scaler_params['std']
        
        # Replace any remaining NaN or Inf with 0
        features_norm = np.nan_to_num(features_norm, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features_norm
    
    def process_all_patients(self, data_dir: str, max_patients: int = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Process all patient files in a directory
        
        Args:
            data_dir: Directory containing .psv files
            max_patients: Maximum number of patients to process (None for all)
            
        Returns:
            List of (features, labels) tuples for each patient
        """
        data_dir = Path(data_dir)
        psv_files = list(data_dir.glob('*.psv'))
        
        if max_patients:
            psv_files = psv_files[:max_patients]
        
        patient_data = []
        
        print(f"Processing {len(psv_files)} patient files...")
        
        for i, psv_file in enumerate(psv_files):
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(psv_files)} patients...")
            
            # Load and clean
            df = self.load_patient_data(str(psv_file))
            if df is None or df.empty:
                continue
            
            df_clean = self.clean_data(df)
            if df_clean is None or df_clean.empty:
                continue
            
            # Create labels
            df_labeled = self.create_labels(df_clean)
            
            # Extract features and labels
            features, labels = self.extract_features(df_labeled)
            
            if features is not None and len(features) > 0:
                patient_data.append((features, labels))
        
        print(f"Successfully processed {len(patient_data)} patients")
        return patient_data
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature column names"""
        return self.feature_columns if self.feature_columns else []

