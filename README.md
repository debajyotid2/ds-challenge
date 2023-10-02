# Code for a Data Science challenge

This repository holds my code submission for a data science challenge.

# Setup

To install all required Python libraries, please ensure to have `Python 3.11.5` installed. Then run

```
git clone https://github.com/debajyotid2/ds-challenge.git
cd ds-challenge
python -m venv venv
```

to clone the repository and create a virtual environment.

If on Windows, run

```
venv\Scripts\activate.bat   # For cmd.exe
venv\Scripts\Activate.ps1   # For PowerShell
```

If on a Mac or Unix-based OS, run

```
source venv/bin/activate
```

These commands will activate the virtual environment on your machine. Finally, run

```
pip install -r requirements.txt
```

to install all dependencies.

# Model training

Assuming the training data is available at `data/historical.txt`, to train all available classification models with their default parameters, please run

```
python main.py --training_data_path data/historical.txt
```

Running `python main.py -h` will show a list of configurable options.

The script trains the classification models with default parameters (no hyperparameter optimization), evaluates them on the validation set, determines the best model based on the ROC-AUC score on the validation set and evaluates the best model on the test set.

Providing training data is mandatory for the script to run.

# Inference

Assuming the data for inference is available at `data/snapshot.txt`, inference can be performed using

```
python main.py --training_data_path data/historical.txt --inference_data_path data/snapshot.txt
```
