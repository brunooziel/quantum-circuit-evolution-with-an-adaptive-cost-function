# Quantum Circuit Evolution with Adaptive Cost Function

This repository contains the implementation of a novel approach to quantum circuit optimization using an adaptive cost function (ACF) framework. The project demonstrates the potential of Quantum Circuit Evolution (QCE) in solving binary optimization problems, particularly in comparison to the Quantum Approximate Optimization Algorithm (QAOA).

## Overview

The project introduces a framework that combines Quantum Circuit Evolution with an Adaptive Cost Function (QCE-ACF) to solve binary optimization problems. This approach offers several advantages:

- Classical optimizer-free methodology
- Dynamic cost function adaptation
- Improved convergence performance
- Shorter execution time compared to QAOA
- Robustness in noisy quantum environments

## Project Structure

- `qce_acf.py`: Main implementation of the Quantum Circuit Evolution with Adaptive Cost Function algorithm
- `_utility.py`: Utility functions and helper methods
- `notebook.ipynb`: Jupyter notebook containing the main implementation and examples
- `instances_benchmark/`: Directory containing benchmark instances for testing

## Reference

1. Franken, L., et al. "Quantum Circuit Evolution on NISQ Devices," 2022 IEEE Congress on Evolutionary Computation (CEC), Padua, Italy, 2022, pp. 1-8. Available at: [10.1109/CEC55065.2022.9870269](https://doi.org/10.1109/CEC55065.2022.9870269).
2. Cacao, R., Cortez, L. R. C. T., & Forner, J. et al. "The Set Partitioning Problem in a Quantum Context," Optimization Letters, vol. 18, pp. 1–17, 2024. Available at: [10.1007/s11590-023-02029-1](https://doi.org/10.1007/s11590-023-02029-1).
3. Fernandez, B. O., Bloot, R., and Moret, M. A., "Quantum Circuit Evolutionary Framework Applied on Set Partitioning Problem," (In review at Springer).
4. Fernandez, B. O., Bloot, R., and Moret, M. A., "Avoiding Convergence Stagnation in a Quantum Circuit Evolutionary Framework Through an Adaptive Cost Function," (In review at IEEE Quantum Week 2025).