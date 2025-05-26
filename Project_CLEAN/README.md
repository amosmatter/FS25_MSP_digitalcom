
# Project CLEAN: Communication Link Estimation & Adaptive Noise-reduction

## Objective
In this project, you will recover a clean signal from noisy observations using LMS and RLS adaptive filters. You’ll compare their convergence speed and noise reduction performance.

## Core Concepts
- Random noise and stochastic processes: Gaussian noise.
- Adaptive filters: LMS, RLS.
- Orthogonality principle and MMSE.
- Practical trade-offs between complexity and performance.

## Tasks
1. **Model the Noisy Signal**
   - Create a clean reference signal (e.g., sine wave or known pattern).
   - Add white Gaussian noise using PySDR’s tools.

2. **Adaptive Filtering with LMS and RLS**
   - Implement LMS and RLS filters to reduce noise in the received signal.
   - Visualize convergence of the error signal.

3. **Compare Performance**
   - Plot error convergence for both filters.
   - Discuss trade-offs in speed and final error.

4. **Analysis and Reflection**
   - Summarize the practical implications of LMS vs. RLS in communication systems.

## Tools
- **PySDR** library (https://pysdr.org) for noise generation and spectrum analysis.
- Custom Python functions (in `/src/utils.py`).

## Deliverables
- **Jupyter Notebook:** complete with:
  - Plots: noisy vs. clean signal, error convergence.
  - Clear explanations of results and theory.
- **Powerpoint presentation:** (15' presentation)

## Evaluation Rubric
| Criteria                                  | Points |  
|-------------------------------------------|--------|  
| Noise modeling and plots                  | 10     |  
| LMS and RLS implementation                | 10     |  
| Convergence plots and analysis            | 10     |  
| Comparison of LMS and RLS                 | 10     |  
| Clarity of explanations and code          | 10     |  
| Final presentation                        | 10     |  
| **Total**                                 | **60** |  

## Instructions
- Work in the provided Jupyter Notebook in `/notebook/`.
- Use PySDR functions and/or extend the helper functions in `/src/utils.py`.
- Save your final notebook as `Project2_CLEAN_YourName.ipynb`.

Enjoy exploring adaptive filters with PySDR!
