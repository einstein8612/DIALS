docker run --network host \
    -it -v $(pwd)/saved_runs/:/saved_runs \
    -v $(pwd)/recurrent_policies/:/recurrent_policies \
    -v $(pwd)/simulators/:/simulators \
    -v $(pwd)/configs/:/configs \
    -v $(pwd)/influence/:/influence \
    -v $(pwd)/plots/:/plots \
    -v $(pwd)/scripts/:/scripts \
    -v $(pwd)/experiment.py:/experiment.py \
    -v $(pwd)/trainer.py:/trainer.py \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY \
    dials python experiment.py with ./configs/traffic/DIALS.yaml