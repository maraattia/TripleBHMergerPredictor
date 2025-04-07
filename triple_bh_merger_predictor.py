#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#--------------------------------------------------------------------------------------------
#--- Implementation of the neural network of Attia & Sibony (2025).
#--------------------------------------------------------------------------------------------
#--- Author: Mara Attia
#--- Date: 07/04/2025
#--------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pickle
import os

class TripleBHMergerPredictor:
    """
    A class for predicting mergers in hierarchical triple black hole systems
    using a pretrained neural network model.
    
    This model was trained on ~15 million simulations performed with a modified 
    version of the JADE secular code that includes gravitational wave emission.
    
    Parameters:
    -----------
    model_path : str
        Path to the pretrained model (keras format)
    scaler_path : str, optional
        Path to the saved scaler (.pkl). If None, will attempt to initialize a 
        new scaler based on parameter ranges.
    """
    
    def __init__(self, model_path, scaler_path=None):
        """Initialize the predictor with a pre-trained model and scaler."""
        # Try to load model
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            raise
        
        self.param_names = ['M1', 'M2', 'M3', 'a_i', 'a_o', 'e_o', 'i_mut']
        self.param_ranges = {
            'M1': [5, 100],       # [M☉]
            'M2': [5, 100],       # [M☉]
            'M3': [1, 200],       # [M☉]
            'a_i': [1, 200],      # [AU]
            'a_o': [100, 10000],  # [AU]
            'e_o': [0, 0.9],
            'i_mut': [40, 80]     # [degree]
        }
        
        # Try to load saved scaler if path is provided
        if scaler_path is not None:
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"Successfully loaded scaler from {scaler_path}")
            except Exception as e:
                print(f"Error loading scaler from {scaler_path}: {e}")
                print("Falling back to default scaler initialization")
                self._initialize_default_scaler()
        else:
            print("No scaler provided or found, initializing default scaler")
            self._initialize_default_scaler()
    
    def _initialize_default_scaler(self):
        """Initialize a default scaler based on parameter ranges."""
        self.scaler = MinMaxScaler()
        
        # Create sample data spanning the parameter ranges
        # This is a simplistic approach and may not match the scaler used during training
        sample_data = []
        for param in self.param_names:
            min_val, max_val = self.param_ranges[param]
            # Add points across the range
            sample_data.append(np.linspace(min_val, max_val, 100))
        
        # Transpose to get [n_samples, n_features]
        sample_data = np.array(sample_data).T
        
        # Fit the scaler
        self.scaler.fit(sample_data)
        print("Initialized default scaler based on parameter ranges")
    
    def predict(self, params, debug=False):
        """
        Predict merger probability for a single system or array of systems.
        
        Parameters:
        -----------
        params : array-like
            System parameters [M1, M2, M3, a_i, a_o, e_o, i_mut]
            or array of such parameters for multiple systems
        
        Returns:
        --------
        probabilities : array
            Merger probabilities (0 to 1)
        """
        if len(np.array(params).shape) == 1:
            # Single system
            params = np.array(params).reshape(1, -1)
        
        # Scale the parameters
        scaled_params = self.scaler.transform(params)
        
        # Add debug info for first few predictions
        if debug:
            if len(params) <= 5:
                print("\nDebug: First few predictions:")
                for i in range(len(params)):
                    print(f"  Original params: {params[i]}")
                    print(f"  Scaled params: {scaled_params[i]}")
                    raw_pred = self.model.predict(scaled_params[i:i+1], verbose=0)[0][0]
                    print(f"  Raw prediction: {raw_pred}")
                    print(f"  Thresholded: {'Merger' if raw_pred >= 0.5 else 'No Merger'}")
                    print("")
        
        # Make predictions
        return self.model.predict(scaled_params, verbose=0).flatten()
    
    def get_confidence(self, probabilities):
        """
        Calculate confidence measure for predictions.
        
        Parameters:
        -----------
        probabilities : array
            Merger probabilities from predict()
        
        Returns:
        --------
        confidence : array
            Confidence measure (0 to 1)
        """
        return 2 * np.abs(probabilities - 0.5)
    
    def evaluate(self, X_test, y_test, debug=False):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : array-like
            Test data features
        y_test : array-like
            Test data labels (0 or 1)
        
        Returns:
        --------
        metrics : dict
            Dictionary with performance metrics
        """
        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        y_prob = self.model.predict(X_test_scaled, verbose=0).flatten()
        
        # Debug: analyze prediction distribution
        if debug:
            print("\nPrediction Distribution Analysis:")
            print(f"  Min prediction: {np.min(y_prob):.6f}")
            print(f"  Max prediction: {np.max(y_prob):.6f}")
            print(f"  Mean prediction: {np.mean(y_prob):.6f}")
            print(f"  Median prediction: {np.median(y_prob):.6f}")
        
            # Count predictions in ranges
            ranges = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                      (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
            
            print("\nPrediction value distribution:")
            for start, end in ranges:
                count = np.sum((y_prob >= start) & (y_prob < end))
                percentage = 100 * count / len(y_prob)
                print(f"  {start:.1f}-{end:.1f}: {count} ({percentage:.2f}%)")
        
        # Apply threshold
        y_pred = (y_prob >= 0.5).astype(int)
        if debug:
            print(f"\nPositive predictions: {np.sum(y_pred)} ({100*np.mean(y_pred):.2f}%)")
        
        confidence = self.get_confidence(y_prob)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Handle edge cases where denominator could be zero
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan  # Precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # Recall/Sensitivity
        tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan  # Specificity
        f1 = 2 * (ppv * tpr) / (ppv + tpr) if (ppv + tpr) > 0 else np.nan
        
        # Calculate high-confidence accuracy
        high_conf_mask = confidence > 0.9
        if np.sum(high_conf_mask) > 0:
            high_conf_acc = np.mean(y_pred[high_conf_mask] == y_test[high_conf_mask])
            high_conf_percentage = np.mean(high_conf_mask) * 100
        else:
            high_conf_acc = np.nan
            high_conf_percentage = 0
        
        # Collect into categories for visualization
        true_pos  = (y_test == 1) & (y_pred == 1)
        true_neg  = (y_test == 0) & (y_pred == 0)
        false_pos = (y_test == 0) & (y_pred == 1)
        false_neg = (y_test == 1) & (y_pred == 0)
        
        # Collect metrics
        metrics = {
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'precision': ppv,
            'recall': tpr,
            'specificity': tnr,
            'npv': npv,
            'f1_score': f1,
            'high_conf_accuracy': high_conf_acc,
            'high_conf_percentage': high_conf_percentage,
            'probabilities': y_prob,
            'confidence': confidence,
            'predictions': y_pred,
            'categories': {
                'true_pos': true_pos,
                'true_neg': true_neg,
                'false_pos': false_pos,
                'false_neg': false_neg
            }
        }
        
        return metrics
    
    def print_metrics(self, metrics):
        """
        Print performance metrics in a formatted way.
        
        Parameters:
        -----------
        metrics : dict
            Output from evaluate()
        """
        print("=" * 50)
        print("TRIPLE BLACK HOLE MERGER PREDICTOR PERFORMANCE")
        print("=" * 50)
        
        # Extract values from confusion matrix
        cm = metrics['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        
        print(f"Test set size: {np.sum(cm)}")
        print(f"Merging systems: {tp + fn} ({(tp + fn) / np.sum(cm):.2%})")
        print(f"Non-merging systems: {tn + fp} ({(tn + fp) / np.sum(cm):.2%})")
        print("\nCONFUSION MATRIX:")
        print(f"{'':15} | {'Predicted NO':15} | {'Predicted YES':15}")
        print("-" * 51)
        print(f"{'Actual NO':15} | {tn:15} | {fp:15}")
        print(f"{'Actual YES':15} | {fn:15} | {tp:15}")
        
        print("\nPERFORMANCE METRICS:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (PPV): {metrics['precision']:.4f}")
        print(f"Recall (TPR): {metrics['recall']:.4f}")
        print(f"Specificity (TNR): {metrics['specificity']:.4f}")
        print(f"Negative Predictive Value (NPV): {metrics['npv']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        print(f"\nHigh confidence predictions (c > 0.9): {metrics['high_conf_percentage']:.2f}%")
        print(f"Accuracy for high confidence predictions: {metrics['high_conf_accuracy']:.4f}")
        
        # Calculate class-specific high confidence metrics
        confidence = metrics['confidence']
        categories = metrics['categories']
        
        high_conf = confidence > 0.9
        high_conf_tp = np.sum(categories['true_pos']  & high_conf)
        high_conf_tn = np.sum(categories['true_neg']  & high_conf)
        high_conf_fp = np.sum(categories['false_pos'] & high_conf)
        high_conf_fn = np.sum(categories['false_neg'] & high_conf)
        
        print("\nHIGH CONFIDENCE METRICS (c > 0.9):")
        if high_conf_tp + high_conf_fp > 0:
            high_conf_ppv = high_conf_tp / (high_conf_tp + high_conf_fp)
            print(f"High confidence precision (PPV): {high_conf_ppv:.4f}")
        
        if high_conf_tn + high_conf_fn > 0:
            high_conf_npv = high_conf_tn / (high_conf_tn + high_conf_fn)
            print(f"High confidence NPV: {high_conf_npv:.4f}")
        
        # Calculate percentage of each category with high confidence
        if np.sum(categories['true_pos']) > 0:
            pct_high_conf_tp = high_conf_tp / np.sum(categories['true_pos']) * 100
            print(f"Percentage of true positives with high confidence: {pct_high_conf_tp:.2f}%")
        
        if np.sum(categories['true_neg']) > 0:
            pct_high_conf_tn = high_conf_tn / np.sum(categories['true_neg']) * 100
            print(f"Percentage of true negatives with high confidence: {pct_high_conf_tn:.2f}%")
        
        if np.sum(categories['false_pos']) > 0:
            pct_high_conf_fp = high_conf_fp / np.sum(categories['false_pos']) * 100
            print(f"Percentage of false positives with high confidence: {pct_high_conf_fp:.2f}%")
        
        if np.sum(categories['false_neg']) > 0:
            pct_high_conf_fn = high_conf_fn / np.sum(categories['false_neg']) * 100
            print(f"Percentage of false negatives with high confidence: {pct_high_conf_fn:.2f}%")
            
        print("=" * 50)

    def predict_system(self, M1, M2, M3, a_i, a_o, e_o, i_mut, verbose=True, debug=False):
        """
        Make prediction for a single system with interpretable output.
        
        Parameters:
        -----------
        M1 : float
            Mass of first black hole in inner binary (solar masses)
        M2 : float
            Mass of second black hole in inner binary (solar masses)
        M3 : float
            Mass of outer black hole (solar masses)
        a_i : float
            Semi-major axis of inner orbit (AU)
        a_o : float
            Semi-major axis of outer orbit (AU)
        e_o : float
            Eccentricity of outer orbit
        i_mut : float
            Mutual inclination between orbits (degrees)
        verbose : bool
            If True, print detailed prediction results
            
        Returns:
        --------
        dict
            Dictionary with prediction results
        """
        # Check parameter constraints
        if M2 > M1:
            if verbose:
                print("Warning: Swapping M1 and M2 to maintain M1 >= M2 convention")
            M1, M2 = M2, M1
        
        if a_o < 10 * a_i:
            if verbose:
                print(f"Warning: a_o ({a_o} AU) < 10 * a_i ({a_i} AU). "
                     "Hierarchical approximation may not be valid.")
        
        # Check that parameters are within the training range
        for name, value in zip(self.param_names, [M1, M2, M3, a_i, a_o, e_o, i_mut]):
            if value < self.param_ranges[name][0] or value > self.param_ranges[name][1]:
                if verbose:
                    print(f"Warning: {name} = {value} is outside the training range "
                         f"[{self.param_ranges[name][0]}, {self.param_ranges[name][1]}]")
        
        # Make prediction
        params = np.array([M1, M2, M3, a_i, a_o, e_o, i_mut]).reshape(1, -1)
        prob = self.predict(params, debug=debug)[0]
        confidence = self.get_confidence(prob)
        
        # Interpret result
        prediction = "Merger" if prob >= 0.5 else "No Merger"
        confidence_level = "Low" if confidence < 0.5 else "Medium" if confidence < 0.9 else "High"
        
        result = {
            'parameters': {
                'M1': M1,
                'M2': M2,
                'M3': M3,
                'a_i': a_i,
                'a_o': a_o,
                'e_o': e_o,
                'i_mut': i_mut
            },
            'prediction': prediction,
            'probability': prob,
            'confidence': confidence,
            'confidence_level': confidence_level
        }
        
        if verbose:
            print("\n===== Triple Black Hole Merger Prediction =====")
            print(f"Inner binary masses: {M1:.1f} M☉, {M2:.1f} M☉")
            print(f"Outer black hole mass: {M3:.1f} M☉")
            print(f"Inner orbit semi-major axis: {a_i:.1f} AU")
            print(f"Outer orbit semi-major axis: {a_o:.1f} AU")
            print(f"Outer orbit eccentricity: {e_o:.3f}")
            print(f"Mutual inclination: {i_mut:.1f}°")
            print(f"\nPrediction: {prediction}")
            print(f"Merger probability: {prob:.3f}")
            print(f"Confidence: {confidence:.3f} ({confidence_level})")
            
            if prediction == "Merger" and confidence_level == "High":
                print("\nThis system will very likely merge within 14 Gyr.")
            elif prediction == "No Merger" and confidence_level == "High":
                print("\nThis system will very likely NOT merge within 14 Gyr.")
            else:
                print("\nThe merger outcome for this system is uncertain.")
            print("================================================")
        
        return result


# Minimal usage example
if __name__ == "__main__":
    # 1. Initialize the predictor with keras model
    model_path  = "model/model_128_128_128_relu.tf"          # Update to your model path
    scaler_path = "model/model_128_128_128_relu_scaler.pkl"  # Update to your scaler path
    
    predictor = TripleBHMergerPredictor(model_path, scaler_path)
    
    # 2. Example system that should result in a merger
    result = predictor.predict_system(
        M1=90, M2=10, M3=100, 
        a_i=50, a_o=1000, 
        e_o=0.8, i_mut=64
    )
    
    # 3. Example system that should not result in a merger
    result = predictor.predict_system(
        M1=30, M2=30, M3=30, 
        a_i=150, a_o=9000, 
        e_o=0.1, i_mut=45
    )