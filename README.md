# Optimization Algorithms for Minimizing Kowalik Function

This repository contains MATLAB code for comparing various optimization algorithms in minimizing the Kowalik function. The algorithms compared include:

1. Newton-Raphson
2. Hestenes-Stiefel
3. Polak-Ribiere
4. Fletcher-Reeves
5. Gradient Descent

## Description

The Kowalik function is a benchmark optimization problem defined as:

\[ f(x) = \sum_{i=1}^{11} \left( a_i - \frac{x_1 (1 + x_2 b_i)}{1 + x_3 b_i + x_4 b_i^2} \right)^2 \]

where the constants \(a_i\) and \(b_i\) are:

\[ a = [0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0
