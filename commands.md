# Commands

## To build

You can also run build.sh

```docker build -t dials .```

## To run

This command uses nushell syntax. Please use run.sh in a linux environment

```docker run -it -v $"(pwd | str replace --all '\' '/')/recurrent_policies/:/recurrent_policies" -v $"(pwd | str replace --all '\' '/')/simulators/:/simulators" -v $"(pwd | str replace --all '\' '/')/configs/:/configs" -v $"(pwd | str replace --all '\' '/')/influence/:/influence" -v $"(pwd | str replace --all '\' '/')/plots/:/plots" -v $"(pwd | str replace --all '\' '/')/scripts/:/scripts" -v $"(pwd | str replace --all '\' '/')/experiment.py:/experiment.py" -v $"(pwd | str replace --all '\' '/')/trainer.py:/trainer.py" dials python experiment.py with ./configs/traffic/DIALS.yaml```
