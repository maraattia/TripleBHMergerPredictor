#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#--------------------------------------------------------------------------------------------
#--- Example usage of the TripleBHMergerPredictor class
#--------------------------------------------------------------------------------------------
#--- This script demonstrates how to use the neural network model
#--- from Attia & Sibony (2025) to predict mergers in hierarchical 
#--- triple black hole systems.
#--------------------------------------------------------------------------------------------
#--- Author: Mara Attia
#--- Date: 07/04/2025
#--------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from triple_bh_merger_predictor import TripleBHMergerPredictor

# Set random seed for reproducibility
np.random.seed(42)

# Labels for the plots
labels_dict = {'M1':'$M_1$ [$M_\\odot$]','M2':'$M_2$ [$M_\\odot$]','M3':'$M_3$ [$M_\\odot$]',
               'a_i':'$a_{\\rm inner}$ [AU]', 'a_o':'$a_{\\rm outer}$ [AU]',
               'e_o':'$e_{\\rm outer}$','i_mut':'$i_{\\rm mut}$[$^\\circ$]'}

def setup_dummy_model_files(model_dir="model"):
    """
    Create dummy model files for testing if they don't exist.
    
    In a real scenario, you would have the actual trained model files.
    This function is just for demonstration purposes.
    
    Parameters:
    -----------
    model_dir : str, optional
        Directory where model files will be stored
    
    Returns:
    --------
    tuple
        Paths to the model and scaler files
    """
    import pickle
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "model_128_128_128_relu.tf")
    scaler_path = os.path.join(model_dir, "model_128_128_128_relu_scaler.pkl")
    
    # Check if model already exists
    if not os.path.exists(model_path):
        print("Creating dummy TensorFlow model...")
        
        # Define a simple 3-layer neural network with 128 neurons per layer
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(7,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
        
        # Save the model
        model.save(model_path)
        print(f"Dummy model saved to {model_path}")
    
    # Check if scaler already exists
    if not os.path.exists(scaler_path):
        print("Creating dummy scaler...")
        
        # Create a scaler with ranges matching those in the TripleBHMergerPredictor class
        scaler = MinMaxScaler()
        
        # Sample data spanning parameter ranges
        param_ranges = {
            'M1': [5, 100],       # [M☉]
            'M2': [5, 100],       # [M☉]
            'M3': [1, 200],       # [M☉]
            'a_i': [1, 200],      # [AU]
            'a_o': [100, 10000],  # [AU]
            'e_o': [0, 0.9],
            'i_mut': [40, 80]     # [degree]
        }
        
        # Create sample data spanning the parameter ranges
        sample_data = []
        for param, (min_val, max_val) in param_ranges.items():
            sample_data.append(np.linspace(min_val, max_val, 100))
        
        # Transpose to get [n_samples, n_features]
        sample_data = np.array(sample_data).T
        
        # Fit the scaler
        scaler.fit(sample_data)
        
        # Save the scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Dummy scaler saved to {scaler_path}")
    
    return model_path, scaler_path

def generate_sample_data(n_samples=100):
    """
    Generate sample data for testing the predictor.
    
    Parameters:
    -----------
    n_samples : int, optional
        Number of samples to generate
    
    Returns:
    --------
    numpy.ndarray
        Array of shape (n_samples, 7) containing sample system parameters
    """
    # Parameter ranges
    param_ranges = {
        'M1': [5, 100],       # [M☉]
        'M2': [5, 100],       # [M☉]
        'M3': [1, 200],       # [M☉]
        'a_i': [1, 200],      # [AU]
        'a_o': [100, 10000],  # [AU]
        'e_o': [0, 0.9],
        'i_mut': [40, 80]     # [degree]
    }
    
    # Generate random samples within parameter ranges
    params = np.zeros((n_samples, 7))
    
    for i, (param, (min_val, max_val)) in enumerate(param_ranges.items()):
        params[:, i] = np.random.uniform(min_val, max_val, n_samples)
    
    # Ensure M2 <= M1 (constraint from the paper)
    mask = params[:, 1] > params[:, 0]  # where M2 > M1
    params[mask, 0], params[mask, 1] = params[mask, 1], params[mask, 0]  # swap M1 and M2
    
    # Ensure a_o > 10 * a_i (hierarchical approximation)
    a_i = params[:, 3]
    a_o = params[:, 4]
    min_a_o = 10 * a_i
    mask = a_o < min_a_o
    params[mask, 4] = min_a_o[mask] + np.random.uniform(0, 100, np.sum(mask))
    
    return params

def example_single_prediction(predictor):
    """
    Demonstrate prediction for a single system.
    
    Parameters:
    -----------
    predictor : TripleBHMergerPredictor
        Initialized predictor object
    """
    print("\n" + "="*80)
    print("Example 1: Prediction for a Single System")
    print("="*80)
    
    # Example system #1 - likely merger based on the paper
    # Close inner orbit, large mass asymmetry, high eccentricity, optimal inclination
    print("\nSystem #1 - Expected to merge:")
    result = predictor.predict_system(
        M1=90, M2=10, M3=100, 
        a_i=50, a_o=1000, 
        e_o=0.8, i_mut=64,
        verbose=True  # Display detailed prediction results
    )
    
    # Example system #2 - unlikely merger based on the paper
    # Equal masses, large separation, low eccentricity, low inclination
    print("\nSystem #2 - Expected NOT to merge:")
    result = predictor.predict_system(
        M1=30, M2=30, M3=30, 
        a_i=150, a_o=9000, 
        e_o=0.1, i_mut=45,
        verbose=True  # Display detailed prediction results
    )
    
    # Example system #3 - uncertain system on the boundary
    print("\nSystem #3 - Uncertain system (boundary case):")
    result = predictor.predict_system(
        M1=82, M2=54, M3=130, 
        a_i=140, a_o=2800, 
        e_o=0.75, i_mut=53,
        verbose=True,  # Display detailed prediction results
    )

def example_batch_prediction(predictor, n_samples=100):
    """
    Demonstrate batch prediction for multiple systems.
    
    Parameters:
    -----------
    predictor : TripleBHMergerPredictor
        Initialized predictor object
    n_samples : int, optional
        Number of samples to generate for batch prediction
    """
    print("\n" + "="*80)
    print("Example 2: Batch Prediction for Multiple Systems")
    print("="*80)
    
    # Generate random sample data
    print(f"\nGenerating {n_samples} random systems...")
    params = generate_sample_data(n_samples)
    
    # Make batch predictions
    print("Making batch predictions...")
    probabilities = predictor.predict(params)
    confidences = predictor.get_confidence(probabilities)
    
    # Process results
    predictions = (probabilities >= 0.5).astype(int)
    merger_count = np.sum(predictions)
    high_conf_count = np.sum(confidences > 0.9)
    
    # Create a DataFrame for better visualization
    results_df = pd.DataFrame({
        'M1': params[:, 0],
        'M2': params[:, 1],
        'M3': params[:, 2],
        'a_i': params[:, 3],
        'a_o': params[:, 4],
        'e_o': params[:, 5],
        'i_mut': params[:, 6],
        'Probability': probabilities,
        'Confidence': confidences,
        'Prediction': ['Merger' if p else 'No Merger' for p in predictions]
    })
    
    # Display summary statistics
    print(f"\nProcessed {n_samples} systems:")
    print(f"- Predicted mergers: {merger_count} ({merger_count/n_samples:.1%})")
    print(f"- Predicted nonmergers: {n_samples - merger_count} ({(n_samples - merger_count)/n_samples:.1%})")
    print(f"- High confidence predictions (c > 0.9): {high_conf_count} ({high_conf_count/n_samples:.1%})")
    
    # Display a few examples
    print("\nSample results (5 random systems):")
    print(results_df.sample(5).to_string(index=False))
    
    return results_df

def visualize_predictions(results_df):
    """
    Visualize the prediction results in relation to parameters.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing prediction results
    """
    print("\n" + "="*80)
    print("Example 3: Visualizing Prediction Results")
    print("="*80)
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Triple Black Hole Merger Predictions', fontsize=16)
    
    # Plot 1: Inner vs Outer separation colored by merger probability
    ax = axs[0, 0]
    scatter = ax.scatter(results_df['a_i'], results_df['a_o'], 
                         c=results_df['Probability'], cmap='viridis', 
                         alpha=0.7, s=50)
    ax.set_xlabel(labels_dict['a_i'])
    ax.set_ylabel(labels_dict['a_o'])
    ax.set_title('Inner vs Outer Separation')
    ax.set_xscale('log')
    ax.set_yscale('log')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Merger Probability')
    
    # Plot 2: Mutual inclination vs Outer eccentricity
    ax = axs[0, 1]
    scatter = ax.scatter(results_df['e_o'], results_df['i_mut'], 
                         c=results_df['Probability'], cmap='viridis', 
                         alpha=0.7, s=50)
    ax.set_xlabel(labels_dict['e_o'])
    ax.set_ylabel(labels_dict['i_mut'])
    ax.set_title('Eccentricity vs Inclination')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Merger Probability')
    
    # Plot 3: Inner masses (M1 vs M2)
    ax = axs[1, 0]
    scatter = ax.scatter(results_df['M1'], results_df['M2'], 
                         c=results_df['Probability'], cmap='viridis', 
                         alpha=0.7, s=50)
    ax.set_xlabel(labels_dict['M1'])
    ax.set_ylabel(labels_dict['M2'])
    ax.set_title('Inner Binary Masses')
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.5)  # Diagonal line M1=M2
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Merger Probability')
    
    # Plot 4: Confidence distribution
    ax = axs[1, 1]
    ax.hist(results_df['Confidence'], bins=20, color='blue', alpha=0.7)
    ax.set_xlabel('Prediction Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Prediction Confidence')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the figure
    plt.savefig('prediction_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to 'prediction_visualization.png'")
    
    # Show the figure
    plt.show()

def example_model_evaluation(predictor):
    """
    Demonstrate model evaluation on test data.
    
    Parameters:
    -----------
    predictor : TripleBHMergerPredictor
        Initialized predictor object
    """
    print("\n" + "="*80)
    print("Example 4: Model Evaluation on Test Data")
    print("="*80)
    
    print("\nLoading test data files...")
    
    # Load actual test data from files
    test_data_dir = os.path.join("model")
    X_test_path = os.path.join(test_data_dir, "X_test.npy")
    y_test_path = os.path.join(test_data_dir, "y_test.npy")
    
    if os.path.exists(X_test_path) and os.path.exists(y_test_path):
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        print(f"Loaded {len(X_test)} test systems with {int(np.sum(y_test))} mergers and {int(len(y_test) - np.sum(y_test))} nonmergers")
    else:
        print(f"Test data files not found at {X_test_path} and {y_test_path}")
        print("Creating sample test data (for demonstration purposes only)...")
        # Create a small test dataset for demonstration
        X_test = generate_sample_data(500)
        y_test = np.random.randint(0, 2, size=500)
        print(f"Created {len(X_test)} test systems with {int(np.sum(y_test))} mergers and {int(len(y_test) - np.sum(y_test))} nonmergers")
    
    # Evaluate the model
    print("Evaluating model performance...")
    metrics = predictor.evaluate(X_test, y_test, debug=True)
    
    # Print results
    predictor.print_metrics(metrics)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = metrics['confusion_matrix']
    
    # Create a heatmap manually using imshow
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)
    
    # Add text annotations for each cell in the confusion matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # Set tick labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Merger', 'Merger'])
    ax.set_yticklabels(['No Merger', 'Merger'])
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to 'confusion_matrix.png'")
    plt.show()
    
    # Create visualization of confidence vs accuracy
    plt.figure(figsize=(10, 6))
    
    # Extract data for plot
    y_prob = metrics['probabilities']
    y_pred = metrics['predictions']
    confidence = metrics['confidence']
    
    # Create bins for confidence
    bins = np.linspace(0, 1, 21)  # 20 bins
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate accuracy in each bin
    accuracies = []
    counts = []
    
    for i in range(len(bins) - 1):
        mask = (confidence >= bins[i]) & (confidence < bins[i+1])
        if np.sum(mask) > 0:
            acc = np.mean(y_pred[mask] == y_test[mask])
            count = np.sum(mask)
        else:
            acc = 0
            count = 0
        accuracies.append(acc)
        counts.append(count)
    
    # Plot accuracy vs confidence
    ax1 = plt.gca()
    ax1.plot(bin_centers, accuracies, 'o-', color='blue', label='Accuracy')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 1.05)
    
    # Plot histogram of confidence values
    ax2 = ax1.twinx()
    ax2.bar(bin_centers, counts, width=(bins[1]-bins[0])*0.8, alpha=0.3, color='gray')
    ax2.set_ylabel('Number of samples', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    plt.title('Accuracy vs Confidence')
    plt.tight_layout()
    plt.savefig('accuracy_vs_confidence.png', dpi=300, bbox_inches='tight')
    print("Accuracy vs. confidence plot saved to 'accuracy_vs_confidence.png'")
    plt.show()

def parameter_space_exploration(predictor, n_points=10):
    """
    Explore how predictions change across the parameter space.
    
    Parameters:
    -----------
    predictor : TripleBHMergerPredictor
        Initialized predictor object
    n_points : int, optional
        Number of points to sample in each dimension
    """
    print("\n" + "="*80)
    print("Example 5: Parameter Space Exploration")
    print("="*80)
    
    # Define base parameters (taken from a system likely to merge)
    base_params = {
        'M1': 68, 
        'M2': 17, 
        'M3': 92, 
        'a_i': 88, 
        'a_o': 6820, 
        'e_o': 0.67, 
        'i_mut': 78.3
    }
    
    print(f"\nBase system parameters: {base_params}")
    print("Exploring how varying each parameter affects merger probability...")
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Effect of Parameters on Merger Probability', fontsize=16)
    axs = axs.flatten()
    
    # Define parameter ranges to explore
    param_ranges = {
        'M1': np.linspace(5, 100, n_points),
        'M2': np.linspace(5, min(base_params['M1'], 100), n_points),
        'M3': np.linspace(1, 200, n_points),
        'a_i': np.logspace(0, 2.3, n_points),  # 1 to 200 AU, log scale
        'a_o': np.logspace(2, 4, n_points),    # 100 to 10,000 AU, log scale
        'e_o': np.linspace(0, 0.9, n_points),
        'i_mut': np.linspace(40, 80, n_points)
    }
    
    # For each parameter, create a batch of systems varying only that parameter
    for i, (param, values) in enumerate(param_ranges.items()):
        if i >= 7:  # We only have 7 parameters
            break
            
        systems = []
        for val in values:
            # Create a copy of base parameters
            system = base_params.copy()
            # Update the parameter being varied
            system[param] = val
            # Convert to list in the expected order
            systems.append([system['M1'], system['M2'], system['M3'], 
                            system['a_i'], system['a_o'], system['e_o'], 
                            system['i_mut']])
        
        # Convert to numpy array and make predictions
        systems = np.array(systems)
        probs = predictor.predict(systems)
        conf = predictor.get_confidence(probs)
        
        # Plot the results
        ax = axs[i]
        ax.plot(values, probs, 'o-', label='Probability')
        
        # Add confidence bands
        ax.fill_between(values, probs - 0.5*(1-conf), probs + 0.5*(1-conf), 
                      alpha=0.2, color='blue', label='Confidence')
        
        ax.set_xlabel(labels_dict[param])
        ax.set_ylabel('Merger Probability')
        
        # Add horizontal line at p=0.5
        ax.axhline(0.5, color='r', linestyle='--', alpha=0.3)
        
        # Set log scale for semi-major axes
        if param in ['a_i', 'a_o']:
            ax.set_xscale('log')
    
    # Remove unused subplot
    if len(param_ranges) < 9:
        for i in range(len(param_ranges), 9):
            fig.delaxes(axs[i])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig('parameter_exploration.png', dpi=300, bbox_inches='tight')
    print("Parameter exploration saved to 'parameter_exploration.png'")
    plt.show()

def main():
    """Main function to run the examples."""
    print("\n" + "="*80)
    print("TRIPLE BLACK HOLE MERGER PREDICTOR EXAMPLE USAGE")
    print("Based on the neural network of Attia & Sibony (2025)")
    print("="*80)
    
    # Set up model files
    print("\nSetting up model files...")
    model_path, scaler_path = setup_dummy_model_files()
    
    # Initialize the predictor
    print("\nInitializing the predictor...")
    predictor = TripleBHMergerPredictor(model_path, scaler_path)
    
    # Run examples
    example_single_prediction(predictor)
    results_df = example_batch_prediction(predictor, n_samples=200)
    visualize_predictions(results_df)
    example_model_evaluation(predictor)
    parameter_space_exploration(predictor)
    
    print("\n" + "="*80)
    print("EXAMPLE USAGE COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()