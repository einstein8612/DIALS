# import sys
# sys.path.append("../..")
from flow.core.params import NetParams
from flow.networks.traffic_light_grid import TrafficLightGridNetwork
from flow.envs import TrafficLightGridBitmapEnv
from flow.core.params import TrafficLightParams
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.controllers import SimCarFollowingController, GridRouter
import numpy as np
import gym 
from gym import spaces

V_ENTER = 10
INNER_LENGTH = 100
LONG_LENGTH = 100
SHORT_LENGTH = 100
N_ROWS = 1
N_COLUMNS = 1
NUM_CARS_LEFT = 0
NUM_CARS_RIGHT = 0
NUM_CARS_TOP = 0
NUM_CARS_BOT = 0
tot_cars = (NUM_CARS_LEFT + NUM_CARS_RIGHT) * N_COLUMNS \
           + (NUM_CARS_BOT + NUM_CARS_TOP) * N_ROWS
grid_array = {
    "short_length": SHORT_LENGTH,
    "inner_length": INNER_LENGTH,
    "long_length": LONG_LENGTH,
    "row_num": N_ROWS,
    "col_num": N_COLUMNS,
    "cars_left": NUM_CARS_LEFT,
    "cars_right": NUM_CARS_RIGHT,
    "cars_top": NUM_CARS_TOP,
    "cars_bot": NUM_CARS_BOT
}
speed_limit = 10
horizontal_lanes = 1
vertical_lanes = 1
program_id = 1
max_gap = 3.0
detector_gap = 0.8
show_detectors = False
phases = [{'duration': '31', 'minDur': '8', 'maxDur': '45', 'state': 'GrGr'},
          {'duration': '6', 'minDur': '3', 'maxDur': '6', 'state': 'yryr'},
          {'duration': '31', 'minDur': '8', 'maxDur': '45', 'state': 'rGrG'},
          {'duration': '6', 'minDur': '3', 'maxDur': '6', 'state': 'ryry'}]

horizon = 300

def gen_edges(col_num, row_num):
    edges = []
    for i in range(col_num):
        edges += ['left' + str(row_num) + '_' + str(i)]
        edges += ['right' + '0' + '_' + str(i)]

    # build the left and then the right edges
    for i in range(row_num):
        edges += ['bot' + str(i) + '_' + '0']
        edges += ['top' + str(i) + '_' + str(col_num)]

    return edges

def get_inflow_params(col_num, row_num, additional_net_params):
    initial = InitialConfig(
        spacing='custom', lanes_distribution=float('inf'), shuffle=True)

    inflow = InFlows()
    outer_edges = gen_edges(col_num, row_num)
    for i in range(len(outer_edges)):
        inflow.add(
            veh_type='idm',
            edge=outer_edges[i],
            probability=0.1,
            depart_lane='free',
            depart_speed=10)

    net = NetParams(
        # osm_path='./osm_path',
        inflows=inflow,
        additional_params=additional_net_params)

    return initial, net

class GlobalTraffic(TrafficLightGridBitmapEnv):

    ACTION_SIZE = 2
    OBS_SIZE = 40
    """
    """
    def __init__(self, seed, learning_agent_ids):
        nodes = []
        for node in range(N_ROWS*N_COLUMNS):
            # FOR TESTING THE ACTUATED TL CONTROLLERS COMMENT THE LINE BELOW
            # if node not in learning_agent_ids:
            nodes.append('center'+str(node))
        additional_env_params = {'target_velocity': 50,
                                'switch_time': 3.0,
                                'num_observed': 2,
                                'discrete': True,
                                'tl_type': 'actuated',
                                'tl_controlled': ['center' + str(id) for id in learning_agent_ids],
                                'scale': 10}
        tl_logic = TrafficLightParams()
        for node in nodes:
            tl_logic.add(node,
                         tls_type='actuated', 
                         programID = program_id, 
                         phases = phases, 
                         maxGap = max_gap, 
                         detectorGap = detector_gap,
                         showDetectors = show_detectors)
        
        additional_net_params = {'grid_array': grid_array,
                                 'speed_limit': speed_limit,
                                 'horizontal_lanes': horizontal_lanes, 
                                 'vertical_lanes': vertical_lanes,
                                 'traffic_lights': True}
        net_params = NetParams(additional_params=additional_net_params)
        vehicles = VehicleParams()
        vehicles.add(veh_id='idm',
                     acceleration_controller=(SimCarFollowingController, {}),
                     car_following_params=SumoCarFollowingParams(
                        min_gap=2.5,
                        decel=7.5,  # avoid collisions at emergency stops
                        max_speed=V_ENTER,
                        speed_mode="all_checks",),
                    routing_controller=(GridRouter, {}),
                    num_vehicles=tot_cars)
        initial_config, net_params = get_inflow_params(col_num=N_COLUMNS,
                                                       row_num=N_ROWS,
                                                       additional_net_params=additional_net_params)
        network = TrafficLightGridNetwork(name='global',
                                          vehicles=vehicles,
                                          net_params=net_params,
                                          initial_config=initial_config,
                                          traffic_lights=tl_logic)
        
        env_params = EnvParams(horizon=horizon, additional_params=additional_env_params)
        sim_params = SumoParams(render=False, restart_instance=False, sim_step=1, print_warnings=False, seed=seed)
        super().__init__(env_params, sim_params, network)
    
    # override
    def reset(self, restart=False):
        if restart:
            self.restart_simulation(self.sim_params)
        states = super().reset()
        observations = []
        for i, node in enumerate(self.tl_controlled):
            node_edges = dict(self.network.node_mapping)[node]
            observation = []
            state = states[i]
            for edge in range(len(node_edges)):
                observation.append(state[edge][:-1]) # last bit is influence source
            observation.append(state[-1]) #  append traffic light info
            observation = np.concatenate(observation)
            observations.append([observation])
        return np.array(observations)

    # override
    def step(self, rl_actions):
        
        # rl_actions = [action.item() for action in rl_actions]
        # rl_actions = int("".join(str(i) for i in rl_actions),2)
        # states, rewards, done, _ = super().step(rl_actions)
        # FOR TESTING THE ACTUATED TL CONTROLLERS COMMENT THE LINE ABOVE AND UNCOMMENT THE LINE BELOW
        states, rewards, done, _ = super().step(None)
        dones = [done]*len(self.tl_controlled)
    
        observations = []
        dsets = []
        infs = []
        for i, node in enumerate(self.tl_controlled):
            node_edges = dict(self.network.node_mapping)[node]
            observation = []
            inf = []
            state = states[i]
            for edge in range(len(node_edges)):
                observation.append(state[edge][:-1])
                inf.append(state[edge][-1]) # last bit is influence source
            dset = np.concatenate(observation) # traffic light info not in dset
            observation.append(state[-1]) #  append traffic light info
            observation = np.concatenate(observation)
            observations.append([observation])
            dsets.append(dset)
            infs.append(inf)
        return np.array(observations), rewards, dones, {'dset': np.array(dsets), 'infs': np.array(infs)}
    
    # override
    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1, shape=(self.OBS_SIZE,))
    
    @property
    def action_space(self):
        """
        Returns A gym dict containing the number of action choices for all the
        agents in the environment
        """
        return spaces.Discrete(self.ACTION_SIZE)

    def _get_influence_sources(self):
        pass

    def close(self):
        self.terminate()

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
