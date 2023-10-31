# DIALS

Source code for the paper [Distributed Influence-Augmented Local Simulators for Parallel MARL in Large Networked Systems](https://openreview.net/forum?id=lKFOwaYNQlb) by Miguel Suau, Jinke He, Matthijs Spaan, and Frans Oliehoek

## Requirements
[Singularity](https://sylabs.io/docs/)

## Installation
```console 
sudo singularity build DIALS.sif DIALS.def
```
This will create a singularity container and install all the required packages. Alternatively, you can create a virtual environment install the packages listed in DIALS.def

## Running an experiment
<strike>
Launch the singularity shell:
```console
singularity shell --writable-tmpfs DIALS.sif
```
To run a new experiment do:
```console
python experiment.py with ./configs/warehouse/DIALS.yaml
```
This will train a new policy for the warehouse environment on DIALS. To train on the global simulator change the config file path to `./configs/warehouse/global.yaml`.
</strike>

Use the interactive tool provided for running any experiments.

Run this tool in bash or as executable
```console
./run.sh
```
or
```console
bash run.sh
```

etc