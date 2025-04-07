[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15167208.svg)](https://doi.org/10.5281/zenodo.15167208)

# Triple Black Hole Merger Predictor

This repository contains the implementation of a neural network model for predicting mergers in hierarchical triple black hole systems, as described in Attia & Sibony (2025). The model was trained on ~15 million secular simulations performed with a modified version of the [JADE](https://github.com/JADE-Exoplanets/JADE) secular code.

## Overview

Hierarchical triple black hole systems consist of an inner binary orbited by a distant third black hole. The gravitational interactions between these objects can induce oscillations in the inner binary's eccentricity through the von Zeipel–Lidov–Kozai (ZLK) mechanism. During periods of high eccentricity, the pericenter distance becomes extremely small, dramatically increasing gravitational wave emission and potentially leading to merger within the age of the Universe.

Our simulations were performed using the [JADE](https://github.com/JADE-Exoplanets/JADE) code, which was originally developed for simulating the secular evolution of hierarchical triple systems in the context of planetary dynamics. We modified JADE to include gravitational wave emission effects, enabling efficient exploration of the triple black hole parameter space.

This neural network allows researchers to quickly predict whether a given triple system configuration will result in a merger of the inner binary within 14 billion years, without requiring full dynamical simulations.

## Repository Structure

- `triple_bh_merger_predictor.py`: Main implementation of the neural network predictor class
- `example_usage.py`: Example script demonstrating all features of the predictor
- `requirements.txt`: List of required Python packages
- `model/`: Directory containing the pretrained model and scaler files
  - `model_128_128_128_relu.tf`: Pretrained neural network model
  - `model_128_128_128_relu_scaler.pkl`: Fitted scaler for input normalization
  - `X_test.npy`: Test dataset features (systems parameters)
  - `y_test.npy`: Test dataset labels (merger outcomes)

## Parameter Ranges

The model is valid for systems with parameters in the following ranges:
- Inner binary masses (`M1`, `M2`): 5 – 100 solar masses (with `M1` ≥ `M2`)
- Outer black hole mass (`M3`): 1 – 200 solar masses
- Inner semi-major axis (`a_i`): 1 – 200 AU
- Outer semi-major axis (`a_o`): 100 – 10,000 AU (with `a_o` > 10 × `a_i`)
- Outer orbit eccentricity (`e_o`): 0 – 0.9
- Mutual inclination (`i_mut`): 40 – 80 degrees

## Requirements

- Python 3.7+
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- scikit-learn

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from triple_bh_merger_predictor import TripleBHMergerPredictor

# Initialize the predictor
model_path = "model/model_128_128_128_relu.tf"
scaler_path = "model/model_128_128_128_relu_scaler.pkl"
predictor = TripleBHMergerPredictor(model_path, scaler_path)

# Predict for a single system
result = predictor.predict_system(
    M1=68, M2=17, M3=92,   # Masses in solar masses
    a_i=88, a_o=6820,      # Semi-major axes in AU
    e_o=0.67, i_mut=78.3   # Eccentricity and mutual inclination (degrees)
)

# The result contains the prediction, probability, and confidence
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.3f}")
print(f"Confidence: {result['confidence']:.3f} ({result['confidence_level']})")
```

### Batch Predictions

```python
import numpy as np

# Create an array of system parameters
# Format: [M1, M2, M3, a_i, a_o, e_o, i_mut]
systems = np.array([
    [68, 17, 92, 88, 6820, 0.67, 78.3],
    [30, 30, 30, 100, 9000, 0.1, 45.0],
    # Add more systems...
])

# Get merger probabilities
probabilities = predictor.predict(systems)

# Get confidence measures
confidences = predictor.get_confidence(probabilities)

# Convert to binary predictions (1 = merger, 0 = no merger)
predictions = (probabilities >= 0.5).astype(int)
```

### Comprehensive Example

For a complete demonstration of all features, run the provided example script:

```bash
python example_usage.py
```

This script demonstrates:
- Single system predictions
- Batch predictions for multiple systems
- Visualization of prediction results
- Model evaluation
- Parameter space exploration

## Model Details

The neural network has the following architecture:
- Input layer (7 neurons): One for each parameter (`M1`, `M2`, `M3`, `a_i`, `a_o`, `e_o`, `i_mut`)
- Three hidden layers (128 neurons each) with ReLU activation
- Output layer (1 neuron) with sigmoid activation

The model achieves 95% overall accuracy, with 99.7% accuracy for high-confidence predictions.

## Citation

If you use this code in your research, please cite:

```
Attia, M., & Sibony, Y. (2025). Exploring the Parameter Space of Hierarchical Triple Black Hole Systems. Astronomy & Astrophysics.
```

If you need to reference this repository, please refer to the [CITATION](CITATION.cff) file.

## License

This code is licensed under the BSD 3-Clause License—see the [LICENSE](LICENSE) file for details.

## Acknowledgements

The contributors to the development of this software are Mara Attia and Yves Sibony. We acknowledge the use of the Claude AI assistant (Anthropic, 2024) for code optimization.

## Contact

For questions or feedback, please contact maraaattia@gmail.com