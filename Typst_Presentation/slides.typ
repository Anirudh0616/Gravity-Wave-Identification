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

#slide[
  = Understanding Wave Parameters

  #cetz.canvas({
    import cetz.draw: *
    plot.plot(size: (3,4), x-tick-step: none, y-tick-step: none, {})
  })

]

#new-section[Methodology]

#slide[
  = Bayesian Statistics

  + *Initialization: * We start with initial parameter values at the midpoints of the given ranges so $ alpha = 1 , beta = 5, gamma = 10 $

  + *Random Walk: * 
  For each iteration we propose a new set of parameters using $ theta_("new") = "normal"( theta_(text("initial")), sigma^2 ) #h(1cm) "where " sigma = [0.01, 0.07, 0.07] $
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


#new-section[Results]

#slide[
  = Covariance Scatter Plots and Histograms
  
  #figure(
    image("MH_corner.png", width: 80%), caption: [Covariance Scatter Plots of Parameters #sym.theta]
  )
]

#slide[
  = Numerical Analysis

  #set table(
  stroke: none,
  gutter: 0.2em,
  fill: (x, y) =>
    if x == 0 or y == 0 { gray },
  inset: (right: 1.5em),
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

  #table(
    columns: 4, 
    [Parameter], [#sym.alpha (alpha)] , [#sym.beta (beta)], [#sym.gamma (gamma)], 
    [Median Value], [1.36], [3.94], [10.00], 
    [95% CI], [0.86 - 1.91], [], []
  )
  
  #set text(size: 0.8em , weight: "light")
]

#slide[
  #show: focus
  Something very important
]
