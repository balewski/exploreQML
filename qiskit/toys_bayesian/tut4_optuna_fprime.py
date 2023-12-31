#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"
'''
Code generated mostly by ChatGPT
Phase 1: Bayesian Optimization - Use Bayesian optimization to find the general area of the minimum.

expand objective function to depends on 2 more variables X ,Y. Let X be a random 2D numpy array of size (nSamp,nFeat), let Y be 1D random array of size (nFeat,)  Let objective function be L2 norm between X*params and Y

'''


import numpy as np
import optuna
import logging
from scipy.optimize import minimize
from functools import partial

# Set Optuna's log level to warning to reduce output
optuna.logging.set_verbosity(optuna.logging.WARNING)
from scipy.optimize import minimize, approx_fprime

def objective_function(params, X, Y):
    # Calculate the difference between X*params and Y
    diff = np.dot(X, params) - Y
    # Return the L2 norm of the difference
    return np.linalg.norm(diff)

def optuna_objective(trial, X, Y):
    # Define the range of parameters for Bayesian Optimization
    params = [trial.suggest_uniform(f'param_{i}', -1, 1) for i in range(X.shape[1])]
    return objective_function(params, X, Y)

def print_callback(study, trial):
    # Print the result and parameters every 10 trials
    if trial.number % 10 == 0:
        print('Optuna Trial=%d  loss=%.3f'%(trial.number,trial.value))
        print(f"Parameters: {trial.params}")

def gradient(params, X, Y):
    # Approximate the gradient of the objective function
    epsilon = np.sqrt(np.finfo(float).eps)
    return approx_fprime(params, lambda p: objective_function(p, X, Y), epsilon)

        
def early_stopping_callback(study, trial, epsilon=1e-3, n_worse_trials=5, n_burnin_trials=10):
    # Initialize or increment the worse_trials_counter
    if 'worse_trials_counter' not in study.user_attrs:
        study.set_user_attr('worse_trials_counter', 0)
    worse_trials_counter = study.user_attrs['worse_trials_counter']

    # Check if this is the best trial so far
    if study.best_value is None or trial.number <= n_burnin_trials:
        return

    if abs(study.best_value - study.best_trial.value) < epsilon:
        # The trial has not improved enough, increment the counter
        worse_trials_counter += 1
        study.set_user_attr('worse_trials_counter', worse_trials_counter)
    else:
        # The trial has improved, reset the counter
        study.set_user_attr('worse_trials_counter', 0)

    # Check if the number of worse trials has exceeded the threshold
    if worse_trials_counter >= n_worse_trials:
        study.set_user_attr('early_stopped', True)
        study.set_user_attr('worse_trials_counter', 0) # reset counter
        print('\n*** Early stop on nTrial=%d, nWorse=%d, eps=%.3f'%(trial.number,worse_trials_counter,epsilon))
       
        study.stop()
        
if __name__ == "__main__":
    # Example usage
    nSamp, nFeat = 100, 5  # Number of samples and features
    X = np.random.rand(nSamp, nFeat)  # Random 2D array (nSamp, nFeat)
    Y = np.random.rand(nSamp)         # Random 1D array (nSamp,)
    
    # Create a study object and specify the optimization direction
    study = optuna.create_study(direction='minimize')
    
    # Use partial to pass additional arguments (X, Y) to the objective function
    study.optimize(partial(optuna_objective, X=X, Y=Y), n_trials=100, callbacks=[print_callback, lambda study, trial: early_stopping_callback(study, trial, 1e-3, 10)])
    # Check if early stopping occurred
    if study.user_attrs.get('early_stopped', False):
        print("M:The Optuna study stopped early due to lack of improvement.\n")
    print("Optimal A loss:", study.best_value)

    # Continue optimization with additional trials
    add_trials = 40  # Define how many more trials you want to run
    study.optimize(partial(optuna_objective, X=X, Y=Y), n_trials=add_trials, callbacks=[print_callback])
    print("Optimal B loss:", study.best_value)
     
    # Get the best parameters found by Optuna
    best_params = [study.best_params[f'param_{i}'] for i in range(nFeat)]
  
    print("Optimal parameters found by Optuna:", best_params)
    print("Optimal loss:", study.best_value)

    # Phase 2: Gradient-Based Refinement using approx_fprime
    param_bounds = [(-1, 1) for _ in range(nFeat)]  # Set bounds for each parameter
    partial_gradient = partial(gradient, X=X, Y=Y)
    result=minimize(fun=lambda p: objective_function(p, X, Y), 
                              x0=list(best_params), 
                              method='L-BFGS-B',  # L-BFGS-B supports bounds
                              jac=partial_gradient, 
                              bounds=param_bounds, 
                              options={'maxiter': 150, 'ftol': 1e-3})
    # When the difference falls below the specified tolerance, the optimization will stop.
    # Output results
    print("Refined optimal parameters found by BFGS:", result.x)
    print("Refined best loss found by BFGS:", result.fun)
