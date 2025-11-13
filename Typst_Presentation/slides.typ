#import "@preview/polylux:0.4.0": *
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
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
    $ theta_("new") ~ N( theta_(text("initial")), sigma^2 ) #h(1cm) "where " sigma = [0.005, 0.081, 0.2] $
    #pagebreak()
    The new value is discarded or chosen based on an _Acceptance Probability_ defined as 
    $ A(theta_("new"), theta_("initial")) = "min"(1 , "Posterior"(theta_("new"))/ "Posterior"(theta_("initial"))) $

    The _Posterior_ function is defined as the following
    $ "Posterior"(theta_"i") = P(theta_"i" | "data") ∝ P("data" | theta_"i") P(theta_"i") $

    #pagebreak()
    Due to the assumptions taken, $P(theta)$ has no effect on the acceptance ratio and our algorithm as a whole.

    $ P(theta) := cases(
  "constant" "if" theta in theta_"constraint",
  0 "everywhere else",
) $

    The only significant metric to consider is now the _Likelihood Function_ $L(theta)$ which we define as

    $
    L(theta) = exp(- 1/(2N) display(sum_i ("y"_"i" - "f"(theta_"i"))^2 / sigma_"i"^2 ))
    $
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
        [Median Value], [1.38], [3.91], [10.00], 
        [95% Credibility Interval], [0.92 - 1.91], [3.63 - 4.19], [9.92 - 10.08],
        [Effective Sample Size], [25.0], [61.1], [1800.0],
        [MC Standard Error], [0.049], [0.019], [0.001]
    )

    The MCMC Algorithm ran with *_Acceptance Ratio_* of _$0.222$_.

    The Global *_Signal to Noise Ratio_* was _$0.97$_, with Local SNR of $0.99$

]

#slide[
    = Measurement Metrics for Metropolis–Hastings

    + *Signal-to-Noise Ratio (SNR):*
    Measures how strongly the true signal stands out from the noise.
    A Global SNR of $0.97$ indicates a moderately clean signal, while
    a Local SNR of $0.99$ shows that the oscillatory region is highly informative.

    + *Effective Sample Size (ESS):*
    MCMC samples are correlated, so the *true* number of independent
    samples is smaller.
    ESS quantifies this:
    $ "ESS" = N_"samples" / (1 + 2 sum_(k=1)^(infinity) rho_"k" ) $
    where $rho_"k"$ is the autocorrelation at lag $k$.

    + *Monte Carlo Standard Error (MCSE):*
    Estimates the uncertainty in the posterior mean *due to sampling noise*.
    It is defined as
    $ "MCSE" = sigma / sqrt("ESS") $
    Low MCSE indicates that the chain produced enough effective samples
    for reliable estimation of parameter statistics.

    + *Autocorrelation:*
    Measures how strongly each MCMC sample depends on earlier samples.
    High autocorrelation means slow exploration and fewer effectively independent samples, directly reducing ESS and increasing MCSE.
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
    #image("Histogram.png")
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

            - *Chosen Scales* ($alpha$: 0.005, $beta$: 0.1, $gamma$: 0.2)
              - _(Roberts & Rosenthal, 2004)_ predicts optimal acceptance ≈ _*0.234*_ for high-dimensional targets.
              - Accuracy peaks — this is the optimal region for efficient sampling.

            - *Large Scales ($>5$)*  
              - Acceptance rate falls → chain stagnates → *accuracy degrades*.
        ]
    )
]

#slide[
    = Likelihood Function
    ->
    #[
        #set text(size: 0.7em)
        The original likelihood makes even tiny parameter changes look catastrophically unlikely, driving acceptance to near zero.
        The new version fixes this by taking a mean instead of summation, keeping acceptance stable.
        #grid(
            columns: (50%, 50%),
            gutter: 0.5em,
            [#image("Likelihood_Comparison.png", width: 100%)],
            [
                ```python
                def likelihood_new(y_data, y_prior):
                    y_err = 0.1 * np.std(y_data)
                    Y = np.mean((y_data-y_prior)**2)/y_err**2
                    return -0.5 * Y

                def likelihood_original(y_datay, y_priory):
                    y_err = 0.1 * (y_data + y_prior) + 1e-6
                    Y = np.sum(((y_data - y_prior)/y_err) ** 2 )
                    return -0.5 * Y
                ```
            ]
        )
    ]
]

#slide[
    = Data Generation for Better Inference
    #grid(
        columns: (2),
        gutter: 1em,
        [
            #set text(size: 0.7em)
            == System Overview
            The data generation pipeline accepts an analytical gravitational wave model function and produces a CSV file containing noisy observations at discrete time points. This system serves two primary purposes:

            - _Algorithm Validation_: Generate data with known ground truth parameters to verify MCMC recovery accuracy
            - _Sensitivity Analysis_: Test algorithm performance under varying noise conditions and proposal scales
            - _Robustness Check_ : Validate algorithm stability under heteroscedastic noise to assess reliability across signal regimes.
        ],
        [
            #set text(size: 0.6em)
            === Noise Formula
            The noise standard deviation for each data point is calculated as:
            ```python
            f_error = noise x f_points + 0.1 x std(f_points)
            f_noisy = f_points + 
                        np.random.normal(0, np.abs(f_error), num)

            ```
            This creates heteroscedastic noise where measurement error grows with signal amplitude.
            === Noise Characteristics

                #table(
                  columns: (auto, auto, auto),
                  align: (left, left, left),
                  stroke: 0.5pt,
                  inset: (x: 8pt, y: 6pt),
                  [*Component*],[*Purpose*],[*Typical Magnitude*],
                  [Proportional ],[Scales with signal amplitude],[Dominant near peaks],
                  [Baseline ],[Constant floor noise],[~4–8 for typical signals],
                )


        ]

    )

]

#slide[
    = Code Structure
    #image("Metropolis_System.png", width: 100%)
]

#new-section[Thank You]

