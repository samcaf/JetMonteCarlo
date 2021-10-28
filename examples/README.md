# Examples

A set of examples which demonstrate the tools of the JetMonteCarlo libraries.

A sample workflow using these tools is shown in [sample_comparison.sh](https://github.com/samcaf/JetMonteCarlo/blob/main/examples/sample_comparison.sh).

## Event Generation
Contains examples of the generation, saving, and loading of events through the generation of phase space emissions either with weights or using parton showers.
Useful for Monte Carlo integration to obtain, for example, distributions of observables.

## Comparison Plots
Contains examples of code for generating plots comparing observable distributions obtained with Monte Carlo integration of weighted phase space emissions and with parton shower algorithms.

## Example workflow
The examples in this folder work together to produce plots which compare distributions of jet energy correlation functions, defined in jetmontecarlo.jets.observables, for analytic calculations in perturbative QCD (pQCD), numerical integration in pQCD, and our parton shower algorithm.

Below is an example bash script that implements this workflow for a particular choice of parameters out of the terminal.
The code below compares quark jets at running coupling and MLL accuracy.
These and other switches may also be adjusted by hand in jetmontecarlo.params.


To run this from the JetMonteCarlo directory, you'll first have to add this library to your python path.
The way I do this is through adding
```
PYTHONPATH="/path/to/JetMonteCarlo:${PYTHONPATH}"
export PYTHONPATH
```
to my ~/.bash_profile file (for mac, or ~/.bashrc for linux), and then using
```
source ~/.bashrc
```

## <a name="license"></a> License

MIT © 2021 Samuel Alipour-fard
