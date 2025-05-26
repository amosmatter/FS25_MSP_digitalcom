## Project CLEAR: Communication Link Evaluation & Analysis with Reception

## Objective
In this project, you will analyze how random noise affects digital communication signals and how an optimal receiver (matched filter) improves system performance.

## Core Concepts
- Random noise and stochastic processes: PDF, PSD.
- Digital modulation: BPSK/QPSK.
- Matched filter as an optimal receiver.

## Tasks
1. **Model the Noise**
   - Generate white Gaussian noise using PySDR and custom Python functions.
   - Plot its PDF and PSD to understand its statistical properties.

2. **Simulate the Digital Communication System**
   - Implement BPSK or QPSK modulation using PySDR’s modulation tools.
   - Transmit the signal through a noisy channel.

3. **Implement a Matched Filter Receiver**
   - Design and apply the matched filter to the received signal.
   - Measure and plot the BER vs. SNR with and without the matched filter.

4. **Analysis and Reflection**
   - Explain how the matched filter improves performance.
   - Discuss how the random noise characteristics (PDF, PSD) impact the BER.

## Tools
- **PySDR** library (https://pysdr.org) for practical, real-world digital communication simulation.
- Custom Python functions (in `/src/utils.py`).

## Deliverables
- **Jupyter Notebook:** complete with:
  - Plots: PDF, PSD, BER vs. SNR, constellation diagrams.
  - Comments and explanations of results.
- **Powerpoint presentation:** (15' presentation)
  
## Evaluation Rubric
| Criteria                              | Points |  
|---------------------------------------|--------|  
| Noise modeling and plots (PDF, PSD)   | 10     |  
| BPSK/QPSK modulation implementation   | 10     |  
| Matched filter design & application   | 10     |  
| BER vs. SNR analysis and plots        | 10     |  
| Clarity of explanations and code      | 10     |  
| Final presentation                    | 10     |  
| **Total**                             | **60** |  

## Instructions
- Work in the provided Jupyter Notebook in `/notebook/`.
- Use PySDR functions and/or extend the helper functions in `/src/utils.py`.
- Save your final notebook as `Project1_CLEAR_YourName.ipynb`.
- Save your final powerpoint as `Project1_CLEAR_YourName.pptx`.

Have a good exploration of digital communication with PySDR!
