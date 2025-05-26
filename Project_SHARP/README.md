
# Project SHARP: Spectral Handling and Adaptive Rejection of Power (Noise)

## Objective
In this project, you will explore how pulse shaping affects the spectrum of digital modulation signals and how it helps suppress out-of-band noise.

## Core Concepts
- PSD of random and modulated signals.
- Digital modulation (e.g., QPSK).
- Linear system impact: root-raised cosine filter.

## Tasks
1. **Generate and Visualize the Modulated Signal**
   - Create a QPSK signal using PySDR.
   - Plot its constellation and time-domain waveform.

2. **Spectral Analysis**
   - Plot the Power Spectral Density (PSD) of the unshaped modulated signal using PySDR’s spectrum tools.

3. **Pulse Shaping with Root-Raised Cosine Filter**
   - Apply an RRC filter to the modulated signal.
   - Plot the new PSD and compare it to the unshaped version.

4. **Analysis and Reflection**
   - Discuss how pulse shaping improves spectral efficiency and suppresses noise.

## Tools
- **PySDR** library (https://pysdr.org) for practical modulation and spectrum analysis.
- Custom Python functions (in `/src/utils.py`).

## Deliverables
- **Jupyter Notebook:** complete with:
  - Plots: constellation, PSD before/after shaping.
  - Clear explanations of results and theory.
- **Powerpoint presentation:** (15' presentation) - some anchor informations
  - How does pulse shaping filter improve real-world digital links?
  - Link to practical systems like LTE or WiFi.

## Evaluation Rubric
| Criteria                                  | Points |  
|-------------------------------------------|--------|  
| Modulated signal and constellation plots  | 10     |  
| PSD analysis before and after shaping     | 20     |  
| RRC filter application and impact         | 10     |  
| Clarity of explanations and code          | 10     |  
| Final reflection and real-world link      | 10     |  
| **Total**                                 | **60** |  

## Instructions
- Work in the provided Jupyter Notebook in `/notebook/`.
- Use PySDR functions and/or extend the helper functions in `/src/utils.py`.
- Save your final notebook as `Project3_SHARP_YourName.ipynb`.

Enjoy exploring spectral shaping in digital communication with PySDR!
