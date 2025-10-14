# Course Setup and Requirements

## Python Environment Setup

### Virtual Environment
It's highly recommended to use a virtual environment for this course:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Required Packages

Create a `requirements.txt` file with the following content:

```txt
# Core packages for scikit-learn best practices
scikit-learn>=1.2.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
ipykernel>=6.0.0

# Additional packages for advanced topics
scipy>=1.7.0
joblib>=1.1.0
plotly>=5.0.0
dash>=2.0.0
flask>=2.0.0
fastapi>=0.70.0
uvicorn>=0.15.0

# Packages for hyperparameter tuning
optuna>=3.0.0
hyperopt>=0.2.5
shap>=0.41.0

# Development and testing tools
pytest>=6.0.0
pytest-cov>=3.0.0
black>=22.0.0
flake8>=4.0.0
```

### Installation Commands

Install all required packages:

```bash
pip install -r requirements.txt
```

For specific modules, you can install additional packages as needed:

```bash
# For deployment modules
pip install flask fastapi uvicorn

# For advanced visualization
pip install plotly dash

# For hyperparameter optimization
pip install optuna hyperopt

# For model interpretability
pip install shap
```

## Dataset Setup

### Sample Datasets

The course uses several standard datasets that are available directly through scikit-learn:

- **Iris Dataset**: For classification examples
- **Boston Housing Dataset**: For regression examples
- **Digits Dataset**: For image classification
- **Wine Dataset**: For multi-class classification
- **Breast Cancer Dataset**: For binary classification

### Loading Datasets

```python
from sklearn import datasets

# Load sample datasets
iris = datasets.load_iris()
boston = datasets.load_boston()
digits = datasets.load_digits()
wine = datasets.load_wine()
breast_cancer = datasets.load_breast_cancer()
```

### Custom Datasets

For exercises and examples, you can use:

```python
# Generate synthetic datasets
from sklearn.datasets import make_classification, make_regression

# Classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42
)

# Regression dataset
X, y = make_regression(
    n_samples=1000,
    n_features=15,
    n_informative=10,
    noise=0.1,
    random_state=42
)
```

## Jupyter Notebook Setup

### Starting Jupyter

```bash
# Start Jupyter notebook
jupyter notebook

# Or start Jupyter lab
jupyter lab
```

### Recommended Extensions

Install these Jupyter extensions for better productivity:

```bash
# Install Jupyter extensions
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# Enable specific extensions
jupyter nbextension enable toc2/main
jupyter nbextension enable execute_time/ExecuteTime
```

## IDE Configuration

### VS Code Setup

If using VS Code, install these extensions:
- Python
- Jupyter
- Pylance
- Python Docstring Generator

### VS Code Settings

Add these settings to your `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "jupyter.askForKernelRestart": false
}
```

## Environment Variables

For consistent reproducibility, set these environment variables:

```bash
# For reproducible results
export PYTHONHASHSEED=0
export SKLEARN_SEED=42

# For parallel processing
export JOBLIB_TEMP_FOLDER=/tmp/joblib
```

## Version Compatibility

This course is designed for:
- Python 3.8+
- scikit-learn 1.2.0+
- NumPy 1.21.0+
- pandas 1.3.0+

## Testing Your Installation

Run this code to verify your installation:

```python
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print(f"scikit-learn version: {sklearn.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"pandas version: {pd.__version__}")

# Test basic functionality
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.3f}")

print("Installation successful! Ready to start the course.")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all packages are installed in the correct environment
2. **Version Conflicts**: Use virtual environments to avoid package conflicts
3. **Memory Issues**: Reduce dataset size or use incremental learning for large datasets
4. **Performance Issues**: Enable parallel processing with `n_jobs=-1`

### Getting Help

- Check the [scikit-learn installation guide](https://scikit-learn.org/stable/install.html)
- Refer to the [troubleshooting section](https://scikit-learn.org/stable/faq.html)
- Ask questions in the course forums or community channels