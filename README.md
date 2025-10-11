# Gravity Wave Identification using Metropolis-Hastings MCMC

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) 

This project implelements a Markov Chain Monte Carlo method -- specifically the Metropolis-Hastings algorithm to identify Gravitational wave-like signals from noisy time series data.
It uses simulated data containing a gravitational wave–like signal with added noise.

![Images to add]()

## Objective
We aim to recover the parameters **α**, **β**, and **γ** of the analytical model that describes the signal:
<p align="center">
</p>
$h(t)=\alpha e^{t\left(1-\tanh!\left(2(t-\beta)\right)\right)}\sin(\gamma t)$

<br>
<br>

* **α** controls the amplitude of the signal  
* **β** shifts the signal in time  
* **γ** controls the oscillation frequency  

<br>

The parameters vary within these ranges:

<p align="center">
</p>
$\quad\quad\quad 0<\alpha<2, \quad 1<\beta<10,\quad 1<\gamma<20$

---

## Method

We perform a random walk in the 3D parameter space:

<p align="center">
</p>
$\quad\quad\quad\quad\quad\quad\quad \theta = (\alpha,\beta,\gamma)$

<br>

<br>


using the **Metropolis–Hastings algorithm** to sample from the posterior probability distribution:

<p align="center">
</p>
$\quad\quad\quad\quad\quad\quad P(\theta | \text{data}) \propto P(\text{data}|\theta)P(\theta)$


<br>
<br>

The prior $P(\theta)$ is uniform within the given ranges.  
The likelihood is calculated as:

<p align="center">
</p>
$\quad\quad\quad\quad\quad\quad P(\text{data}|\theta) \propto \exp(Y)$

<br>
<br>
<p align="left">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;where</p>

<p align="center">
</p>
$\quad\quad\quad\quad\quad\quad Y=-\sum_i\frac{(y_{\text{data},i}-y_{\text{model},i})^2}{y_{\text{err},i}^2}$

<br>
<br>


The error at each data point is assumed to be **20%**. (given)

Each iteration proposes a new set of parameters. If the new parameters give a higher probability, they are accepted; otherwise, they may still be accepted with a probability proportional to their likelihood ratio.  
This helps the algorithm explore the entire parameter space instead of getting stuck in local maxima.

---

## Outputs

The program estimates the most likely values of **α**, **β**, and **γ**, and also provides uncertainty ranges.  

We have plotted the following:
- Posterior distributions for each parameter  
- A reconstructed signal overlaid on the noisy data  
- Histograms or corner plots showing parameter correlations and convergence  

---

## Concepts Used

- Metropolis–Hastings algorithm (MCMC sampling)  
- Bayesian inference  
- Parameter estimation under noise  
- Gravitational wave signal modeling  

---
## Project Structure
```text
Gravity-Wave-Identification/
├── Data_Gen/               # Mock gravitational wave data
├── Source/                 # MCMC and plotting scripts
├── Configurations/         # Configure Functions and Parameters 
└── requirements.txt        # Python dependencies
```
---
## How to Run
1. Clone the Repository
```bash
   git clone https://github.com/Anirudh0616/Gravity-Wave-Identification.git
   cd Gravity-Wave-Identification
```
2. Install the dependencies (maybe in a virtual environment!)
```bash
    pip3 install -r requirements.txt
```
3. Run the main scripts
```bash
    python3 src/mcmc_gravity_wave.py
```

(You might have to use python/pip instead of python3/pip3)

---
## References

- Metropolis et al. (1953), Equation of State Calculations by Fast Computing Machines
- Hastings (1970), Monte Carlo Sampling Methods Using Markov Chains
- Gregory, P. C. (2005), Bayesian Logical Data Analysis for the Physical Sciences
 - [StatLect MCMC Notes](https://www.statlect.com/fundamentals-of-statistics/Metropolis-Hastings-algorithm)
---
### Authors
This repository is our collective project for Computational Physics EP4210 at IIT Hyderabad (Fall 2025)


[Anirudh Bhat](https://github.com/Anirudh0616)  
[Samyak Rai](https://github.com/Sammybro11)  
[Shanmukh Machiraju](https://github.com/1mach0)

