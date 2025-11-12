# Gravity Wave Identification using Metropolis-Hastings MCMC

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) 

This project implements a Markov Chain Monte Carlo method -- specifically the Metropolis-Hastings algorithm to identify Gravitational wave-like signals from noisy time series data.
It uses simulated data containing a gravitational wave–like signal with added noise.

![Predicted Fit](https://github.com/Anirudh0616/Gravity-Wave-Identification/blob/main/Results/Original_Unknown/Gravitational_Wave_pred.png)

## Objective
We aim to recover the parameters **α**, **β**, and **γ** of the analytical model that describes the signal:

```math
h(t)=\alpha e^{t}\left(1-\tanh\left(2(t-\beta)\right)\right)\sin(\gamma t)
```

* **α** controls the amplitude of the signal  
* **β** shifts the signal in time  
* **γ** controls the oscillation frequency  

<br>
The parameters vary within these ranges:


```math
0<\alpha<2, \quad 1<\beta<10,\quad 1<\gamma<20
```

---

## Method

We perform a random walk in the 3D parameter space:
```math
\theta = (\alpha,\beta,\gamma)
```

<br>

using the **Metropolis–Hastings algorithm** to sample from the posterior probability distribution:

```math
P(\theta | \text{data}) \propto P(\text{data}|\theta)P(\theta)
```

<br>

The prior $P(\theta)$ is uniform within the given ranges.  
The likelihood is calculated as:
```math
P(\text{data}|\theta) \propto \exp(Y)
```

<br>

where

```math
Y=-\frac{1}{2 N}\sum_i^N \left(\frac{(y_{\text{data},i}-y_{\text{model},i})^2}{y_{\text{err},i}^2}\right)
```
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

### Histrograms and Trace Plots
![Histogram](https://github.com/Anirudh0616/Gravity-Wave-Identification/blob/main/Results/Original_Unknown/Histogram.png)

### Parameter Covariance Plots

| Alpha vs Beta                                                                                                                              | Beta vs Gamma                                                                                                                              | Gamma vs Alpha                                                                                                                               |
|--------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| ![AlphaVsBeta](https://github.com/Anirudh0616/Gravity-Wave-Identification/blob/main/Results/Original_Unknown/Covariance/Alpha_vs_Beta.png) | ![BetaVsGamma](https://github.com/Anirudh0616/Gravity-Wave-Identification/blob/main/Results/Original_Unknown/Covariance/Beta_vs_Gamma.png) | ![GammaVsAlpha](https://github.com/Anirudh0616/Gravity-Wave-Identification/blob/main/Results/Original_Unknown/Covariance/Gamma_vs_Alpha.png) |

---
### Results
The algorithm ran with an acceptance rate of $22.2$%

The predicted parameters are:

| Parameter    | Alpha $\alpha$    | Beta $\beta$      | Gamma $\gamma$     |
|--------------|-------------------|-------------------|--------------------|
| Median Value | $1.38$            | $3.91  $          | $10.00$            |
| 95% Interval | $\(0.92 - 1.91\)$ | $\(3.64 - 4.19\)$ | $\(9.92 - 10.08\)$ |

The model graph using the predicted parameters is shown in the image above and `Results/Original_Unknown/Gravitational_Wave_pred.png`

## Concepts Used

- Metropolis–Hastings algorithm (MCMC sampling)  
- Bayesian inference  
- Parameter estimation under noise  
- Gravitational wave signal modeling  

---
## Project Structure
```text
Gravity-Wave-Identification/
├── main.py                     # Main orchestrator with mode routing
├── run.sh                      # Bash launcher for interactive execution
├── gw_data.csv                 # Unknown gravitational wave data (problem input)
├── requirements.txt            # Python dependencies (numpy, pandas, matplotlib, etc.)
├── README.md                   # Repository documentation
│
├── Configurations/             # Configuration and model definitions
│   ├── Grav_Wave.yaml          # MCMC parameters for generated/unknown modes
│   ├── variance_test.yaml      # MCMC parameters for variance testing
│   └── gw_functions.py         # Gravitation_Wave class, likelihood functions
│
├── Source/                     # Core algorithm and visualization
│   ├── metropolis_hasting.py   # MetroHaste MCMC implementation
│   └── plotting.py             # Visualization functions (histogram_gw, corner_plot, etc.)
│
├── Data_Gen/                   # Data generation pipeline
│   ├── generator.py            # Generate_Data function
│   └── Data_Grav_Wave.csv      # Synthetic noisy data (generated mode output)
│
├── Results/                    # Output directory for experiments
│   ├── [experiment_id]/        # Custom experiment outputs (generated mode)
│   │   ├── Gravitational_Wave_data.png
│   │   ├── Gravitational_Wave_pred.png
│   │   ├── MH_hist.png
│   │   └── Covariance/         # Alpha_vs_Beta.png, Beta_vs_Gamma.png, Gamma_vs_Alpha.png
│   ├── Original_Unknown/       # Unknown mode outputs
│   │   ├── Gravitational_Wave_pred.png
│   │   ├── Histogram.png
│   │   └── Covariance/         # Alpha_vs_Beta.png, Beta_vs_Gamma.png, Gamma_vs_Alpha.png
│   └── Variance_Test.png       # Variance mode diagnostic plot
│   └── Likelih_Comparison.png  # Comparison between Likelihood Functions
│
├── Typst_Presentation/         # Documentation and presentation
│   └── slides.typ              # Typst source for scientific presentation
│
└── manim_animation/            # Parameter effect animations
```
---
## How to Run
1. Clone the Repository
```bash
   git clone https://github.com/Anirudh0616/Gravity-Wave-Identification.git
   cd Gravity-Wave-Identification
```
2. Create and Activate Virtual Environment
```bash
# MacOS/Linux
python3 -m venv .venv
source .venv/bin/activate
# Windows ( Command Prompt )
py -3 -m venv .venv
..venv\Scripts\activate.bat
```
3. Install the dependencies
```bash
pip install -r requirements.txt
```
4. Run the Launcher
```bash
# MacOS/Linux
chmod +x run.sh
./run.sh
# Windows
bash run.sh
```
The Launcher will prompt you to enter values for test experiment and experiment name ( Directory name for plots to be saved in )

5. Deactivate afterwards
```bash
deactivate
```
---
## References

- Metropolis et al. (1953), Equation of State Calculations by Fast Computing Machines
- Hastings (1970), Monte Carlo Sampling Methods Using Markov Chains
- Gregory, P. C. (2005), Bayesian Logical Data Analysis for the Physical Sciences
- Charles Zaiontz, Effective Sample Size for Metropolis Algorithm
- Gareth O. Roberts, Jeffrey S. Rosenthal "General state space Markov chains and MCMC algorithms," Probability Surveys, Probab. Surveys 1(none), 20-71, (2004) 
 - Taboga, Marco (2021). "Metropolis-Hastings algorithm", Lectures on probability theory and mathematical statistics. Kindle Direct Publishing. Online appendix.
[StatLect MCMC Notes](https://www.statlect.com/fundamentals-of-statistics/Metropolis-Hastings-algorithm)

---
### Authors
This repository is our collective project for **Dr. Kirit Makwana's** Computational Physics EP4210 at IIT Hyderabad (Fall 2025)


[Anirudh Bhat](https://github.com/Anirudh0616) -- [Samyak Rai](https://github.com/Sammybro11) -- [Shanmukh Machiraju](https://github.com/1mach0)

