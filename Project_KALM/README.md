# Project KALM: Kalman-based Adaptive Linear Modeling

## Objective
In this project, you will simulate a time-varying random signal (e.g., random walk + noise) and use a Kalman filter to estimate the clean signal in real time.

## Core Concepts
- Stochastic processes (random walks, dynamic signals).
- Kalman filter for optimal linear estimation (MMSE).
- Linear system impact on noisy measurements.

## Tasks
1. **Simulate a Time-Varying Noisy Signal**
   - Create a random walk or other time-varying random process.
   - Add Gaussian noise using PySDR’s tools or custom functions.

2. **Implement the Kalman Filter**
   - Use a basic Kalman filter to estimate the clean signal.
   - Visualize filter tracking and error convergence.

3. **Analysis and Reflection**
   - Discuss the filter’s ability to track the dynamic signal.
   - Link to real-world applications (e.g., tracking in wireless or sensor systems).

## Tools
- **PySDR** Examples (https://pysdr.org) for noise generation and analysis.
Examples avaialble at : https://github.com/777arc/PySDR/tree/ba4f470767bbd5b460217ec2f78ade82845c15e8/figure-generating-scripts
- Custom Python functions (in `/src/utils.py`).

## Deliverables
- **Jupyter Notebook:** complete with:
  - Plots: noisy vs. estimated signals, error convergence.
  - Clear explanations of results and theory.
- **Powerpoint presentation:** (15' presentation) - 
  - How does the Kalman filter adapt to the signal?
  - Practical uses in communication and sensor systems.

## Evaluation Rubric
| Criteria                                   | Points |  
|--------------------------------------------|--------|  
| Noisy signal simulation and plots          | 10     |  
| Kalman filter implementation and tracking  | 20     |  
| Error convergence visualization            | 10     |  
| Clarity of explanations and code           | 10     |  
| Final presentation                         | 10     |  
| **Total**                                  | **60** |  

## Instructions
- Work in the provided Jupyter Notebook in `/notebook/`.
- Use PySDR functions and/or extend the helper functions in `/src/utils.py`.
- Save your final notebook as `Project4_KALM_YourName.ipynb`.

Enjoy exploring real-time adaptive filtering with the Kalman filter and examples of PySDR!
