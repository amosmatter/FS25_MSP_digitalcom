# Project DEFI: Decision-Feedback Equalizer Implementation

## Objective
In this project, you will simulate a digital link with inter-symbol interference (ISI) and design a basic Decision Feedback Equalizer (DFE) to mitigate it.

## Core Concepts
- ISI effects in digital channels.
- DFE as an optimal receiver extension.
- Eye diagrams and BER as performance metrics.

## Tasks
1. **Simulate an ISI Channel**
   - Create a simple 2-tap ISI channel model.
   - Generate a digital signal (e.g., BPSK) and pass it through the ISI channel.

2. **Visualize ISI Effects**
   - Plot eye diagrams and constellation plots of the signal before equalization.

3. **Implement DFE Equalizer**
   - Design and apply a DFE to reduce ISI-induced errors.
   - Visualize the eye diagram and constellation after equalization.

4. **Analysis and Reflection**
   - Compute BER before and after equalization.
   - Discuss how DFE improves reliability in real channels.

## Tools
- **PySDR** Examples available (https://pysdr.org) for digital signal visualization and simulation.
- Custom Python functions (in `/src/utils.py`).

## Deliverables
- **Jupyter Notebook:** complete with:
  - Plots: eye diagrams, constellation plots, BER comparison.
  - Clear explanations of results and theory.
- **Powerpoint presentation:** (15' presentation) 
  - How does DFE mitigate ISI?
  - Link to practical communication systems (e.g., DSL, LTE).

## Evaluation Rubric
| Criteria                                  | Points |  
|-------------------------------------------|--------|  
| ISI channel simulation and visualization  | 10     |  
| DFE implementation and visualization      | 20     |  
| BER analysis before and after             | 10     |  
| Clarity of explanations and code          | 10     |  
| Final presentation                        | 10     |  
| **Total**                                 | **60** |  

## Instructions
- Work in the provided Jupyter Notebook in `/notebook/`.
- Use PySDR functions and/or extend the helper functions in `/src/utils.py`.
- Save your final notebook as `Project5_DEFI_YourName.ipynb`.

Explore how DFE makes digital communication more robust in real-world systems!
