# PRISMA

PRISMA: Physics-informed Regression and Inference for Statistics-driven Measurement Acquisition

## Description

PRISMA is a Bayesian active learning algorithm that automates and optimizes
spectroscopy measurements. PRISMA uses prior parameter distributions and known
physics model equations to optimize measurement acquisition and fit model
parameters in real-time. Active learning increases the speed and accuracy of
spectroscopy measurements while maintaining confidence and rigor.

## Authors and acknowledgment

Developed by Dante Basile, Xavier Tricoche, Sam Lilak, Devika Mehta, Luca Basso, Andy Mounce, and Ezra Bussmann.

This software was developed as part of work at the CINT (Center for Integrated Nanotechnologies), an Office of Science User Facility operated for the U.S. Department of Energy.

## Installation

Install in editable mode for development:

```bash
python3 -m pip install -e .
```

Install from `requirements.txt` for reproducible environment:

```bash
python3 -m pip install -r requirements.txt
```

## Usage

Examples of PRISMA-directed spectroscopy experiments can be found in the
`experiments` folder. These Jupyter notebooks show how to use
PRISMA with both pseudorandom simulated data and instrument measurements.

As a general overview, PRISMA defines the Bayesian online nonlinear regression
algorithm in the `ActiveParameterFit` abstract base class in
`src/prisma/apf/active_parameter_fit.py`. Subclasses implement this interface
by implementing the `acquisition_f` abstract method with their acquisition
function of choice to define their sampling strategy. For example,
`ActiveParameterFitSigmaSqPred` samples at the maximum of the predictive
variance and can be imported in the API at
`prisma.apf.ActiveParameterFitSigmaSqPred`.

Sampler classes in `src/prisma/samplers` provide drivers that allow PRISMA to acquire measurements
at the domain values it specifies.

Call the `fit` method of an initialized `ActiveParameterFit` instance to
run PRISMA's active learning algorithm. This will optimize measurement
acquisitions and fit model parameters.

## License

[BSD-3-Clause](https://opensource.org/license/bsd-3-clause)

## Citation

```bibtex
@software{prisma2026github,
  author = {Dante Basile and Xavier Tricoche and Sam Lilak and Devika Mehta and Luca Basso and Andy Mounce and Ezra Bussmann},
  title = {{PRISMA}: {P}hysics-informed {R}egression and {I}nference for {S}tatistics-driven {M}easurement {A}cquisition},
  url = {http://github.com/},
  version = {0.0.1},
  year = {2026},
}
```
