# Quantum Dropout for Land Use and Land Cover Classification

## Overview

This research project explores an innovative quantum machine learning approach to Land Use and Land Cover (LULC) classification using a novel quantum dropout technique applied to satellite imagery.

## Key Features

- Implements a quantum dropout layer for neural network regularization
- Uses EuroSAT satellite image dataset with 10 land cover classes
- Applies quantum circuit-based data encoding for dropout probability determination
- Compares performance with classical dropout and no dropout approaches

## Research Methodology

### Data
- **Dataset**: EuroSAT Satellite Image Dataset [EuroSAT](https://ieeexplore.ieee.org/document/8736785)
- **Image Size**: 64×64 pixels
- **Spectral Bands**: RGB
- **Total Images**: 27,000
- **Classes**: 10 different land cover types
- **Train/Test Split**: 80:20

### Model Architecture

- **Backbone**: ResNet-50
- **Classification Head**: Fully connected network
- **Regularization Techniques**:
    1. No dropout
    2. Classical dropout (p=0.5)
    3. Quantum dropout (σ=0.1)

### Quantum Dropout Approach

- Data-encoding circuit with Z-rotation gates, √X gates, and CNOT gates
- 8-qubit quantum circuit
- Measurement-based dropout probability determination
- Uses Jensen-Shannon divergence to analyze measurement distributions

### Key Contributions

- First exploration of quantum dropout for LULC classification
- Analysis of quantum circuit's ability to distinguish image classes
- Insights into quantum machine learning for remote sensing applications

### Results Highlights

- Demonstrated regularization potential of quantum dropout
- Identified challenges in quantum circuit design for classification tasks
- Showed promise for adaptive, data-driven dropout mechanisms

### Future Work

- Develop more sophisticated quantum circuit designs
- Explore dynamic adaptation of quantum dropout
- Investigate computational optimization strategies

### License

- [MIT](LICENSE)

### Acknowledgments

This project was supported by members from the Institut Quantique and the Département de géomatique appliquée at the Université de Sherbrooke including Dr. Marco Armenta, Tania Belabbas, Dr. Samuel Foucher, and others.

## References

[1] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 2019.