#import "@preview/polylux:0.4.0": *
#import "@preview/metropolis-polylux:0.1.0" as metropolis
#import metropolis: new-section, focus
#import "@preview/cetz:0.4.2"
#import "@preview/cetz-plot:0.1.3": plot, chart

#show: metropolis.setup

#slide[
    #set page(header: none, footer: none, margin: 3em)

    #text(size: 1.3em)[
        *Gravitation Wave Identification*
    ]

    Using Metropolis Hastings Algorithm

    #metropolis.divider

    #set text(size: .8em, weight: "light")
    Group 1

    #set text(size: .8em, weight: "extralight")
    Anirudh Bhat

    Samyak Rai

    Shanmukh Machiraju
]

#slide[
    = Index

    #metropolis.outline
]

#new-section[Problem Statement]

#slide[
    = Problem Statement

    Given a time series strain data with added noise with the structure of a gravitational wave as given below
    $
    h(t) = alpha e^(t) [1 - tanh{2(t- beta )}] sin(gamma t)
    $

    $alpha , beta , gamma $ are parameters that signify the physical properties of the given wave. There value ranges are
    $
    0 < & alpha < 2 \
    1 < & beta < 10 \
    1 < & gamma < 20 
    $

    We need to determine the parameter values using a _Metropolis Hastings_ Random walk algorithm in the 3 dimensional space.
]

#let wave(t, alpha, beta, gamma) = {
    alpha * exp(t) * (1 - tanh(2 * (t - beta))) * sin(gamma * t)
}

#show link: set text(fill: blue)

#slide[
    = Understanding Wave Parameters

    Let us visualize how the parameters α, β, and γ influence the waveform  
    // $h(t) = α e^t [1 - tanh(2(t - β))] sin(γ t)$.

    + α controls the amplitude of the signal
    + β shifts the signal in time
    + γ controls the oscillation frequency

    *
    #link("https://drive.google.com/file/d/1plsZ-eYz4k4COAh8hJ9uOK-cN_qz1XhB/view")[Animation Paramater effects] *

    //#figure(
    //   image("preview.png", width: 55%)
    // )


]

#new-section[Methodology]

#slide[
    = Random Walks

    + *Initialization: * We start with initial parameter values at the midpoints of the given ranges so $ alpha = 1 , beta = 5, gamma = 10 $

    + *Random Walk: * 
    For each iteration we propose a new set of parameters using 
    $ theta_("new") = "normal"( theta_(text("initial")), sigma^2 ) #h(1cm) "where " sigma = [0.01, 0.07, 0.07] $ 
    // NEED TO CHANGE THIS THE SCALES
    #pagebreak()
    The new value is discarded or chosen based on an _Acceptance Probability_ defined as 
    $ A(theta_("new"), theta_("initial")) = "min"(1 , "Posterior"(theta_("new"))/ "Posterior"(theta_("initial")))
    $

    The _Posterior_ function is defined as the following 
    ```python
    def likelihood_reduced(y_data: np.ndarray, y_prior: np.ndarray):
        y_err = 0.1 * np.std(y_data)
        Y = np.mean((y_data - y_prior) ** 2) / y_err**2
    return -0.5 * Y
    ```

    #set text(size: 0.8em , weight: "light")

    This is different from the function provided in the problem statement, we will explain why this is better in Section 4
]

#slide[
    = Stochastic Maximum Likelihood Estimation
    Hello 
    
]

#slide[
    = Why not Bayesian Inference ?
    Hello

]


#new-section[Results]

#slide[
    = Numerical Analysis

    #set table(
        stroke: none,
        gutter: 0.2em,
        fill: (x, y) =>
        if x == 0 or y == 0 { orange },
        inset: (right: 1.5em, top: 0.5em, bottom: 0.5em),
    )
    #show table.cell: it => {
        if it.x == 0 or it.y == 0 {
            set text(white)
            strong(it)
        } else if it.body == [] {
            // Replace empty cells with 'N/A'
            pad(..it.inset)[_N/A_]
        } else {
            it
        }
    }
    == Parameter Values
    #table(
        columns: 4, 
        [Parameter], [#sym.alpha (alpha)] , [#sym.beta (beta)], [#sym.gamma (gamma)], 
        [Median Value], [1.44], [3.90], [10.00], 
        [95% Credibility Interval], [0.90 - 1.93], [3.61 - 4.19], [9.92 - 10.08],
        [Effective Sample Size], [62.3], [121.7], [800.0],
        [MC Standard Error], [0.036], [0.013], [0.001]
    )

    The MCMC Algorithm ran with *_Acceptance Ratio_* of _$0.263$_.

    The Global *_Signal to Noise Ratio_* was _$1.00$_, with Local SNR of $1.02$

]

#slide[
    = Measurement Metrics for Metropolis Hastings
    Talk about signal to noise ratio, ESS, AFC, MC Std Err
]



// Autocorrelation measures **how much each MCMC sample depends on its predecessors**.
//
// If your chain at step ( i ) is very similar to step ( i-1 ), ( i-2 ), … then it’s *highly autocorrelated* — meaning it’s not exploring new regions quickly.
//
// If the samples are nearly independent (uncorrelated), then you’re sampling efficiently — you’re getting new information every iteration.
//
// In essence:
//
// * **High autocorrelation → redundant samples → slow mixing**
// * **Low autocorrelation → diverse samples → fast mixing**
//
//
// For a single parameter sequence ( {\theta_t}_{t=1}^N ):
//
// [
// \rho_k = \frac{\text{Cov}(\theta_t, \theta_{t+k})}{\text{Var}(\theta_t)}
// ]
//
// * ( \rho_k ): autocorrelation at lag ( k )
// * ( k = 1, 2, 3, \dots ) steps apart
//
// Intuitively, ( \rho_1 ) tells you how correlated consecutive samples are.
// If ( \rho_1 ≈ 1 ), your chain barely moves — think of it as a car idling in traffic.
// If ( \rho_1 ≈ 0 ), your chain jumps around freely — exploration is good.
//
//
// Autocorrelation directly determines **how many effectively independent samples** your chain contains — this is the **Effective Sample Size (ESS)** you’ve been using.
//
// [
// N_{\text{eff}} = \frac{N}{1 + 2\sum_{k=1}^{\infty}\rho_k}
// ]
//
// If autocorrelations decay slowly (stay high even at large lags), the sum is large → ESS is small.
// That means although you ran, say, 10,000 steps, you only have the equivalent of maybe 100 independent samples.
//
// So **autocorrelation = measure of inefficiency**.
//
//
//
// | Behavior                           | Autocorrelation shape                         | Implication                                                  |
// | ---------------------------------- | --------------------------------------------- | ------------------------------------------------------------ |
// | Sharp drop (ρ ≈ 0 after few steps) | Exponential decay                             | Chain mixes well                                             |
// | Slowly decaying tail               | Flat then gradual drop                        | Chain stuck, increase proposal step size or adapt covariance |
// | Alternating +/− pattern            | Oscillatory model or overcorrection proposals | Maybe reduce step size slightly                              |
//
// You already saw:
// [
// MCSE = \frac{s}{\sqrt{ESS}}
// ]
// Since ( ESS ) depends on ( \rho_k ), high autocorrelation → smaller ( ESS ) → larger MCSE → less precise estimate of the mean.
// So autocorrelation is *the fundamental quantity controlling how uncertain your MCMC-based estimates are.*


// 2. The logic of SNR and parameter identifiability
// SNR	Data appearance	Effect on α estimation
// Low (SNR ≲ 1)	waveform barely visible, noise dominates	likelihood surface flat in α → poor identifiability, wide CI, low ESS
// Moderate (SNR ≈ 3–10)	signal visible but noisy	α estimable with some uncertainty
// High (SNR ≫ 10)	waveform dominates noise	α tightly constrained, narrow CI, high ESS
#slide[
    = Prediction vs Data
    #image("Gravitational_Wave_pred.png")
]

#slide[
    = Covariance Scatter Plots 
    #figure(
        grid(
            columns: 3,
            gutter: 2mm,
            [#image("Alpha_vs_Beta.png", width: 100%)],
            [#image("Beta_vs_Gamma.png", width: 100%)],
            [#image("Gamma_vs_Alpha.png", width: 100%)],
        )
    )
]

#slide[
    = Histograms and Trace Plots
    Heloo 
    // Imma do this later figure it out if you can
]

#new-section[Optimization]

#slide[
    = Scale Selection
    #grid(
        columns: 2,
        gutter: 0.5em,
        [#image("Variance_Test.png")],
        [
            #set text(size: 0.7em)
            - *Small Scales ($<10^(-1)$)*  
              - Chain barely moves → strong autocorrelation → *low accuracy*.

            - *Near Chosen Scales*
              - _(Roberts & Rosenthal, 1997)_ predicts optimal acceptance ≈ _*0.234*_ for high-dimensional targets.
              - Accuracy peaks — this is the optimal region for efficient sampling.

            - *Large Scales ($>5$)*  
              - Acceptance rate falls → chain stagnates → *accuracy degrades*.
        ]
    )
]

#slide[
    = Likelihood Function Selection

    Explain why our likelihood function is better
]

#slide[
    = Data Generation for Better Inference
    Hello
]

#slide[
    = Code Structure
    Hello
]

#new-section[Thank You]

