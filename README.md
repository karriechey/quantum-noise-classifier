# Quantum Noise Dataset Generator

## Overview
This project generates synthetic quantum measurement data with realistic noise models using Qiskit Aer. The dataset is designed for training machine learning models to classify quantum noise types.

## Dataset Specification
- **Total samples**: ~10,000 (1,000 circuits × 10 noise classes)
- **Features**: ~20-30 statistical features extracted from quantum measurements
- **Labels**: 10 classes (1 no-noise control + 9 noisy conditions)
- **Noise types**: Depolarizing, Amplitude Damping, Phase Damping
- **Noise intensities**: Low (0.01), Medium (0.05), High (0.1)

## Installation

### Option 1: Local Installation (Recommended)

```bash
# Create a virtual environment (optional but recommended)
python -m venv quantum_env
source quantum_env/bin/activate  # On Windows: quantum_env\Scripts\activate

# Install dependencies
pip install qiskit qiskit-aer pandas numpy tqdm

# Verify installation
python -c "import qiskit; print(f'Qiskit version: {qiskit.__version__}')"
```

### Option 2: Google Colab

```python
# Run this in a Colab cell
!pip install qiskit qiskit-aer pandas numpy tqdm

# Upload the script or copy-paste the code
# Then run the main function
```

### Option 3: Conda

```bash
conda create -n quantum python=3.10
conda activate quantum
pip install qiskit qiskit-aer pandas numpy tqdm
```

## Requirements
- Python 3.8+
- qiskit >= 0.45.0
- qiskit-aer >= 0.13.0
- pandas >= 1.5.0
- numpy >= 1.24.0
- tqdm >= 4.65.0

## Usage

### Basic Usage
```bash
python quantum_noise_dataset_generator.py
```

This will:
1. Generate 10,000 quantum circuits
2. Simulate them with different noise models
3. Extract features from measurement outcomes
4. Save the dataset as CSV and pickle files

### Expected Runtime
- **Laptop/Desktop**: 10-30 minutes
- **Google Colab**: 15-40 minutes
- **High-performance workstation**: 5-15 minutes

The script is CPU-bound (no GPU needed).

### Output Files
After running, you'll get:
1. `quantum_noise_dataset.csv` - Main dataset (human-readable)
2. `quantum_noise_dataset.pkl` - Pickle format (faster loading)
3. `quantum_noise_dataset_summary.txt` - Statistics and metadata

## Dataset Structure

### Columns

**Metadata** (7 columns):
- `circuit_id`: Unique identifier for each circuit
- `label`: Noise class (0-9)
- `noise_name`: Descriptive name of noise condition
- `noise_type`: Type of noise (depolarizing/amplitude_damping/phase_damping/none)
- `noise_intensity`: Noise strength (0, 0.01, 0.05, 0.1)
- `num_qubits`: Number of qubits in circuit (2-5)
- `circuit_depth`: Number of gate layers (5-15)

**Features** (~20-30 columns):
Statistical features extracted from quantum measurements, including:
- Probability distribution statistics (max, min, mean, std)
- Entropy measures (Shannon entropy, normalized entropy)
- Purity metrics (purity, participation ratio)
- Per-qubit marginal probabilities
- Two-point correlations between qubits
- Number of observed states

### Noise Classes

| Label | Noise Type | Intensity | Description |
|-------|-----------|-----------|-------------|
| 0 | None | 0.0 | Control (no noise) |
| 1 | Depolarizing | 0.01 | Low depolarizing |
| 2 | Amplitude Damping | 0.01 | Low amplitude damping |
| 3 | Phase Damping | 0.01 | Low phase damping |
| 4 | Depolarizing | 0.05 | Medium depolarizing |
| 5 | Amplitude Damping | 0.05 | Medium amplitude damping |
| 6 | Phase Damping | 0.05 | Medium phase damping |
| 7 | Depolarizing | 0.1 | High depolarizing |
| 8 | Amplitude Damping | 0.1 | High amplitude damping |
| 9 | Phase Damping | 0.1 | High phase damping |

## Customization

### Modify Dataset Size
Edit these parameters in the script:
```python
CIRCUITS_PER_CLASS = 1000  # Change to generate more/fewer samples
SHOTS = 1024                # Measurements per circuit
```

### Adjust Circuit Complexity
```python
NUM_QUBITS_RANGE = (2, 5)   # Min and max qubits
DEPTH_RANGE = (5, 15)        # Min and max circuit depth
```

### Change Noise Levels
```python
NOISE_LEVELS = {
    'low': 0.01,
    'medium': 0.05,
    'high': 0.1
}
```

## Loading the Dataset

### In Python
```python
import pandas as pd

# Load CSV
df = pd.read_csv('quantum_noise_dataset.csv')

# Or load pickle (faster)
df = pd.read_pickle('quantum_noise_dataset.pkl')

# Separate features and labels
X = df.drop(['circuit_id', 'label', 'noise_name', 'noise_type', 
             'noise_intensity', 'num_qubits', 'circuit_depth'], axis=1)
y = df['label']

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
```

## Next Steps: ML Training

Once you have the dataset, you can train classifiers:

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_pickle('quantum_noise_dataset.pkl')

# Prepare features and labels
feature_cols = [col for col in df.columns if col not in 
                ['circuit_id', 'label', 'noise_name', 'noise_type', 
                 'noise_intensity', 'num_qubits', 'circuit_depth']]
X = df[feature_cols].values
y = df['label'].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Evaluate
train_acc = clf.score(X_train_scaled, y_train)
test_acc = clf.score(X_test_scaled, y_test)

print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
```

## Troubleshooting

### ImportError: No module named 'qiskit'
Solution: `pip install qiskit qiskit-aer`

### Slow execution
- Normal for 10,000 samples
- Reduce `CIRCUITS_PER_CLASS` for faster testing
- Use fewer qubits: `NUM_QUBITS_RANGE = (2, 3)`

### Memory issues
- Reduce `CIRCUITS_PER_CLASS`
- Process in batches
- Close other applications

## Scientific Background

### Why These Noise Models?
1. **Depolarizing**: Most general noise model, represents random errors
2. **Amplitude Damping**: Energy dissipation (T1 relaxation)
3. **Phase Damping**: Dephasing without energy loss (T2 dephasing)

These three cover the main types of errors in real quantum hardware.

### Feature Engineering Rationale
- **Entropy**: Noise increases randomness
- **Purity**: Noise creates mixed states (lower purity)
- **Correlations**: Noise destroys quantum entanglement
- **Marginals**: Different noise types affect qubits differently

## References
- Qiskit Documentation: https://qiskit.org/documentation/
- Qiskit Aer: https://qiskit.org/ecosystem/aer/
- Noise models in quantum computing: Nielsen & Chuang, "Quantum Computation and Quantum Information"

## License
This code is provided as-is for educational and research purposes.

## Contact
For questions or issues, consult Qiskit documentation or quantum computing forums.
