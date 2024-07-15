FROM python:3.6

RUN apt-get update -y
RUN apt-get install -y git cmake python3 g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev python3-dev swig default-jdk maven libeigen3-dev

# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
# RUN dpkg -i cuda-keyring_1.1-1_all.deb
# RUN apt-get update
# RUN apt-get -y install cuda-toolkit-12-1

RUN git clone --recursive https://github.com/eclipse/sumo
RUN export SUMO_HOME="$PWD/sumo"
RUN cmake -B build /sumo
RUN cmake --build build -j$(nproc)

# RUN chmod -R 777 /usr/local/lib/python3.6/site-packages

WORKDIR /

RUN git clone https://github.com/INFLUENCEorg/flow.git
RUN chmod -R 777 /flow/

RUN git clone https://github.com/miguelsuau/recurrent_policies.git
RUN pip install -e ./flow
RUN pip install stable-baselines3
RUN pip install numpy
RUN pip install matplotlib
RUN pip install pyaml
RUN pip install sacred
RUN pip install pymongo
RUN pip install sshtunnel
RUN pip install networkx
RUN pip install lxml
RUN pip install pyglet
RUN pip install imutils
RUN pip install scipy
RUN pip install torch==1.8.1
# RUN pip install https://download.pytorch.org/whl/cu111/torch-1.8.1%2Bcu111-cp36-cp36m-linux_x86_64.whl
RUN pip install pathos
RUN pip install psutil
RUN pip install opencv-python==4.5.3.56

COPY simulators/ ./simulators
COPY configs/ ./configs
# COPY flow/ ./flow
COPY influence/ ./influence
COPY plots/ ./plots
COPY scripts/ ./scripts
COPY experiment.py ./
COPY trainer.py ./
# COPY recurrent_policies ./recurrent_policies
# COPY logs.monitor.csv ./ idk but this line errors

RUN pip install -e ./simulators/warehouse/
RUN pip install -e simulators/traffic/

ENV SUMO_HOME="/sumo"
ENV PYTHONPATH="${SUMO_HOME}/tools:${PYTHONPATH}"
ENV PATH="${SUMO_HOME}/bin:${PATH}"