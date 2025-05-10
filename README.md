# Enhancing Physics-Informed Neural Networks for Solving Stiff ODEs and PDEs

This repository contains the implementation and results corresponding to the Dual Degree Project titled "Enhancing Physics-Informed Neural Networks for Solving Stiff ODEs and PDEs".

### Overview

Physics-Informed Neural Networks (PINNs) are a powerful class of deep learning models that integrate governing physical laws directly into the learning process by incorporating differential equation residuals into the loss function. Despite their success in various scientific applications, standard PINNs exhibit poor performance when solving stiff differential equations—those characterized by wide disparities in time or spatial scales.

This work proposes and evaluates novel strategies to improve the accuracy, convergence, and stability of PINNs on stiff PDEs and ODEs, culminating in a new framework called ATHENA-PINN.

**ATHENA-PINN**: Combines the following:
  - Adaptive sampling based on Hessian, Gradients, and Residuals
  - Time marching through sequential intervals
  - Triangular cyclic sampling schedule for smooth training
  - Second-order optimizers (LBFGS and Newton-CG ([Source](https://github.com/pratikrathore8/opt_for_pinns/blob/main/src/opts/nys_newton_cg.py) for Newton-CG code alone))
  - Gradient-weighted loss scaling

### Repository Structure


├── optimizers/             -> Implementations of 2nd order optimizer  
├── plots/                  -> Output heatmaps and figures  
├── src/                    -> PINN models    
├── .gitignore           
├── requirements.txt    
└── README.md          

