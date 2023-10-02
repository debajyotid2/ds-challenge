# Code for a Data Science challenge

This repository holds my code submission for a data science challenge. The challenge involves predicting the probability of a failure in a machine within a pre-specified forecast window (of X cycles), from the sensor readings of the machine. 

The initial steps of data cleaning and feature engineering involve calculation of statistics like mean and median from the sensor data from a rolling window of observations. The data is also labeled by giving positive labels to the observations within the failure forecast window and negative labels to all other observations. The data is then split by assigning data from randomly selected machine IDs to training, validation and test sets. This is done to ensure that all data from each unique machine is assigned to one and only one subset.

Then different classification models like logistic regression and decision trees are trained and evaluated on the validation subset using metrics like balanced accuracy, precision, recall, F1 score and ROC-AUC. Accuracy is not used because of the class imbalance (significantly large number of negative examples compared to positive). Out of all the models trained, the best model is selected based on the ROC-AUC score on the validation set. 

Finally, the best model is evaluated on the test set and used for final inference (if features for inference are provided).

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
