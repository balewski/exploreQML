#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
'''
Code generated mostly by ChatGPT
Phase 1: Bayesian Optimization - Use Bayesian optimization to find the general area of the minimum.
Phase 2: Gradient-Based Refinement - Use gradient-based methods with approx_fprime near the area found in Phase 1 to refine the solution.
'''

import numpy as np
import optuna
import logging
from scipy.optimize import minimize, approx_fprime

# Set Optuna's log level to warning to reduce output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Define a callback function to print the result every 10 trials
def print_callback(study, trial):
    if trial.number % 10 == 0:
        print('Optuna Trial=%d  loss=%.3f'%(trial.number,trial.value))
        print(f"    Parameters: {trial.params}")

# Define your objective function that runs the experiment and returns the loss
def objective_function(params):
    # Replace this with the actual experiment
    return np.sum(params**2)

def gradient(params):
    epsilon = np.sqrt(np.finfo(float).eps)
    return approx_fprime(params, objective_function, epsilon)

def optuna_objective(trial):
    # Define the range of parameters for Bayesian Optimization
    x1 = trial.suggest_float('x1', -2, 2)
    x2 = trial.suggest_float('x2', -2, 2)
    params = np.array([x1, x2])
    return objective_function(params)

# Create a study object and specify the optimization direction
study = optuna.create_study(direction='minimize')
study.optimize(optuna_objective, n_trials=100, callbacks=[print_callback])  # You can set your trial limit

print('M:Optun end')
for xN in ['x1','x2']:
    print('best %s = %.4f'%(xN,study.best_params[xN]))

# Get the best parameters found by Optuna
x_approx = np.array([study.best_params['x1'], study.best_params['x2']])

# Gradient-Based Refinement using approx_fprime
result = minimize(objective_function, x_approx, method='BFGS', jac=gradient)

print("Approximate minimum found by Optuna:", x_approx)
print("Refined minimum found by BFGS:", result.x)
