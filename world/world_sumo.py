"""
Part of this code is borrowed from RESCO: https://github.com/Pi-Star-Lab/RESCO
"""
import csv

import os
import sys
import time
from math import atan2, pi
import xml.etree.cElementTree as ET
import random
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit('No SUMO in environment path')
from common.registry import Registry

import json
import re
import copy
import pandas as pd
import sumolib
import libsumo
import traci
import networkx as nx
from heapq import heappush, heappop
from itertools import count
import networkx as nx
from heapq import heappush, heappop
from itertools import count

import networkx as nx
from heapq import heappush, heappop
from itertools import count
class Intersection(object):
    '''
    Intersection Class is mainly used for describing crossing information and defining acting methods.
    '''
    def __init__(self, id, world, phases):
        self.id = id
        self.world = world
        self.eng = self.world.eng
        self.lanes = []
        self.roads = []
        self.outs = []
        self.directions = []
        self.out_roads = []
        self.in_roads = []
        self.road_lane_mapping = {}
        self.interface_flag = world.interface_flag

        # map_name = Registry.mapping['world_mapping']['setting'].param['network']
        # self.lane_order_cf = None
        # self.lane_order_sumo = None
        # if 'signal_config' in Registry.mapping['world_mapping']['setting'].param.keys():
        #     if 'N' in Registry.mapping['world_mapping']['setting'].param['signal_config'][map_name]['cf_order'].keys():
        #         self.lane_order_cf = Registry.mapping['world_mapping']['setting'].param['signal_config'][map_name]['cf_order']
        #         self.lane_order_sumo = Registry.mapping['world_mapping']['setting'].param['signal_config'][map_name]['sumo_order']
        #     else:
        #         if self.id in Registry.mapping['world_mapping']['setting'].param['signal_config'][map_name]['cf_order'].keys():
        #             self.lane_order_cf = Registry.mapping['world_mapping']['setting'].param['signal_config'][map_name]['cf_order'][self.id]
        #             self.lane_order_sumo = Registry.mapping['world_mapping']['setting'].param['signal_config'][map_name]['sumo_order'][self.id]
        #         else:
        #             self.lane_order_cf = Registry.mapping['world_mapping']['setting'].param['signal_config'][map_name]['cf_order'][self.id[3:]] # exclude 'GS_'
        #             self.lane_order_sumo = Registry.mapping['world_mapping']['setting'].param['signal_config'][map_name]['sumo_order'][self.id[3:]]

        # links and phase information of each intersection
        self.current_phase = 0
        self.virtual_phase = 0  # see yellow phase as the same phase after changing
        self.next_phase = 0
        self.current_phase_time = 0

        self.yellow_phase_time = min([i.duration for i in self.eng.trafficlight.getAllProgramLogics(self.id)[0].phases])
        self.map_name = world.map  # TODO: try to add it to Registry later

        self.lanelinks = world.eng.trafficlight.getControlledLinks(self.id)
        for link in self.lanelinks:
            # skip if empty link
            if not link:
                continue
            link = link[0]
            if link[0][:-2] not in self.road_lane_mapping.keys():
                self.road_lane_mapping.update({link[0][:-2]: []})  # assume less than 9 lanes in each road
                self.road_lane_mapping[link[0][:-2]].append(link[0])
                self.roads.append(link[0][:-2])
                self.outs.append(False)
                road = self.eng.lane.getShape(link[0])
                self.directions.append(self._get_direction(road, False))
            elif link[0][:-2] in self.road_lane_mapping.keys() and link[0] not in self.road_lane_mapping[link[0][:-2]]:
                self.road_lane_mapping[link[0][:-2]].append(link[0])
            if link[1][:-2] not in self.road_lane_mapping.keys():
                self.road_lane_mapping.update({link[1][:-2]: []})  # assume less than 9 lanes in each road
                self.road_lane_mapping[link[1][:-2]].append(link[1])
                self.roads.append(link[1][:-2])
                self.outs.append(True)
                road = self.eng.lane.getShape(link[1])
                self.directions.append(self._get_direction(road, True))
            elif link[1][:-2] in self.road_lane_mapping.keys() and link[1] not in self.road_lane_mapping[link[1][:-2]]:
                self.road_lane_mapping[link[1][:-2]].append(link[1])

        self._sort_roads()
        for key in self.road_lane_mapping.keys():
            for lane in self.road_lane_mapping[key]:
                self.lanes.append(lane)

        self.green_phases = phases
        self.phases = [i for i in range(len(phases))]
        self.phase_available_startlanes = []
        self.startlanes = []
        self.phase_available_lanelinks = []
        for r, p in enumerate(self.green_phases):
            tmp_lanelinks = []
            tmp_startane = []
            for n, i in enumerate(p.state):
                if i == 'G' or i == 's':
                    # skip if empty link
                    links = self.world.eng.trafficlight.getControlledLinks(self.id)
                    
                    # links = links[n]
                    if n >= len(links):
                        break
                    else:
                        links = links[n]
                                            
                    if not links:
                        continue
                    links = links[0]
                    tmp_lanelinks.append([links[0], links[1]])
                    if links[0] not in tmp_startane:
                        tmp_startane.append(links[0])
                    if links[0] not in self.startlanes:
                        self.startlanes.append(links[0])
            self.phase_available_startlanes.append(tmp_startane)
            self.phase_available_lanelinks.append(tmp_lanelinks)

        self.full_phases, self.yellow_dict = self.create_yellows(self.green_phases, self.yellow_phase_time, self.interface_flag)
        # programs = self.eng.trafficlight.getAllProgramLogics(self.id)
        tl_id = self.id + "_rl"
        logic = self.eng.trafficlight.Logic(tl_id, 0, 0, self.full_phases)
        self.eng.trafficlight.setProgramLogic(self.id, logic)
        self.eng.trafficlight.setProgram(self.id, tl_id)

        # dictionary of remembered features
        self.waiting_times = dict()
        self.full_observation = None
        self.full_observation_part = None
        self.last_step_vehicles = None
        self.reset_runtime_profile()

        # TODO: check .signals .full_observation .last_stet_vehicles need to be set or not

    def _sort_roads(self):
        '''
        _sort_roads
        Sort roads information by arranging an order.
        
        :param: None
        :return: None
        '''
        order = sorted(range(len(self.roads)),
                       key=lambda i: (self.directions[i],
                                      self.outs[i] if self.world.RIGHT else not self.outs[i]))
        self.roads = [self.roads[i] for i in order]
        self.directions = [self.directions[i] for i in order]
        self.outs = [self.outs[i] for i in order]
        self.out_roads = [self.roads[i] for i, x in enumerate(self.outs) if x]
        self.in_roads = [self.roads[i] for i, x in enumerate(self.outs) if not x]  # TODO: check if its 4

    def reset(self):
        '''
        reset
        Reset information, including current_phase, full_observation and last_step_vehicles, etc.
        
        :param: None
        :return: None
        '''
        self.current_phase_time = 0
        self.virtual_phase = 0
        self.next_phase = 0
        self.waiting_times = dict()
        self.full_observation = None
        self.full_observation_part = None
        self.last_step_vehicles = None
        self.current_phase = self.get_current_phase()
        # eng is set in world
        programs = self.eng.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.full_phases
        self.eng.trafficlight.setProgramLogic(self.id, logic)

    def get_current_phase(self):
        '''
        get_current_phase
        Get current phase of current intersection.
        
        :param: None
        :return cur_phase: current phase of current intersection
        '''
        cur_phase = self.eng.trafficlight.getPhase(self.id)
        return cur_phase

    # TODO: change cityflow phase generator into phase property
    def prep_phase(self, new_phase):
        '''
        prep_phase
        Prepare change phase of current intersection

        :param new_phase: phase that will be executed in the later
        :return: None
        '''
        if self.get_current_phase() == new_phase:
            self.next_phase = self.get_current_phase()
            if self.interface_flag:
                self.eng.trafficlight.setPhase(self.id, int(self.next_phase))
            else:
                self.eng.trafficlight.setPhase(self.id, self.next_phase)
            self.current_phase = self.get_current_phase()
        else:
            self.next_phase = new_phase
            # find yellow phase between cur and next phases
            y_key = str(self.get_current_phase()) + '_' + str(new_phase)
            if y_key in self.yellow_dict:
                y_id = self.yellow_dict[y_key]
                if self.interface_flag:
                    self.eng.trafficlight.setPhase(self.id, int(y_id))  # phase turns into yellow here
                else:
                    self.eng.trafficlight.setPhase(self.id, y_id)  # phase turns into yellow here
                self.current_phase = self.get_current_phase()

    def _change_phase(self, phase):
        '''
        _change_phase
        Change phase at current intersection.
        
        :param phase: phase to be executed at the next step
        :return: None
        '''
        if self.interface_flag:
            self.eng.trafficlight.setPhase(self.id, int(phase))
        else:
            self.eng.trafficlight.setPhase(self.id, phase)
        self.current_phase = self.get_current_phase()

    def pseudo_step(self, action):
        '''
        pseudo_step
        Take relative actions and calculate time duration of current phase.
        
        :param action: the changes to take
        :return: None
        '''
        # TODO: check if change state, yellow phase must less than minimum of action time
        # test yellow finished first
        self.virtual_phase = action
        if self.current_phase_time == self.yellow_phase_time:
            self._change_phase(action)
        else:
            if action != self.get_current_phase() and self.current_phase_time > self.yellow_phase_time:
                self.current_phase_time = 0
            if self.current_phase_time == 0:
                self.prep_phase(action)
            elif self.current_phase_time < self.yellow_phase_time:
                self._change_phase(self.current_phase)
            else:
                self._change_phase(action)

        self.current_phase_time += 1




    def reset_runtime_profile(self):
        self.runtime_profile = {
            "visible_csv_read_count": 0,
            "visible_csv_read_wall_s": 0.0,
            "observe_count": 0,
            "observe_wall_s": 0.0,
            "observepart_count": 0,
            "observepart_wall_s": 0.0,
        }

    def get_runtime_profile(self):
        return dict(getattr(self, "runtime_profile", {}))

    def _add_runtime_profile(self, key, elapsed_s):
        profile = getattr(self, "runtime_profile", None)
        if profile is None:
            return
        profile[f"{key}_count"] = profile.get(f"{key}_count", 0) + 1
        profile[f"{key}_wall_s"] = profile.get(f"{key}_wall_s", 0.0) + elapsed_s

    def get_vehicle_ids_set_csv(self, path=None):
        start = time.perf_counter()
        if path is None:
            path = Registry.mapping['command_mapping']['setting'].param.get(
                'omnet_csv_path',
                '/home/exx/Desktop/vtc2026/omnet_files/gwu-workspace-pedestrians/'
                'simu5G/simulations/NR/cars/SUMO_output_CV2X.csv',
            )
        try:
            if not path or not os.path.isfile(path):
                return set()
            # 1. Read the first line to detect delimiter
            with open(path, 'r', newline='') as f:
                header = f.readline().strip()
            delimiter = ';' if ';' in header else ','
            
            # 2. Now open again as a DictReader with the right delimiter
            with open(path, 'r', newline='') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                cols = reader.fieldnames
                if 'VehicleID' not in cols:
                    raise ValueError(f"Column 'VehicleID' not found. Available columns: {cols}")
                
                # 3. Build the set
                vehicle_ids = { row['VehicleID'] for row in reader }
            
            return vehicle_ids
        finally:
            self._add_runtime_profile("visible_csv_read", time.perf_counter() - start)



    def observe(self, step_length, distance):
        start = time.perf_counter()
        '''
        observe
        Get observation of the whole roadnet, including lane_waiting_time_count, lane_waiting_count, lane_count and queue_length.
        
        :param step_length: time duration of step
        :param distance: distance limitation that it can only get vehicles which are within the length of the road
        :return: None
        '''
        use_omnet = Registry.mapping['command_mapping']['setting'].param.get('use_omnet', False)
        if not use_omnet:
            self.full_observation = {}
            self._add_runtime_profile("observe", time.perf_counter() - start)
            return
        penetration_rate=1
        vehicle_ids=self.get_vehicle_ids_set_csv()

        full_observation = dict()
        all_vehicles = set()
        for lane in self.lanes:
            vehicles = []
            lane_measures = {'lane_waiting_time_count': 0, 'lane_waiting_count': 0, 'lane_count': 0, 'queue_length': 0}
            lane_vehicles_all = self._get_vehicles(lane, distance)

            if lane_vehicles_all:
                num_to_select = max(1, int(len(lane_vehicles_all) * penetration_rate)) if penetration_rate > 0 else 0
                lane_vehicles = random.sample(lane_vehicles_all, min(num_to_select, len(lane_vehicles_all)))
            else:
                lane_vehicles = []

            for v in lane_vehicles:
                if v not in vehicle_ids:
                    continue
                all_vehicles.add(v)
                if v in self.waiting_times:
                    self.waiting_times[v] += step_length
                elif self.eng.vehicle.getWaitingTime(v) > 0:
                    self.waiting_times[v] = self.eng.vehicle.getWaitingTime(v)
                v_measures = dict()
                v_measures['name'] = v
                v_measures['wait'] = self.waiting_times[v] if v in self.waiting_times else 0
                #TODO: CHEC ITS RIGHT CALCULATION?
                lane_measures['queue_length'] = lane_measures['queue_length'] + 1
                v_measures['speed'] = self.eng.vehicle.getSpeed(v)
                v_measures['position'] = self.eng.vehicle.getLanePosition(v)
                vehicles.append(v_measures)
                if v_measures['wait'] > 0:
                    lane_measures['lane_waiting_time_count'] += v_measures['wait']
                    lane_measures['lane_waiting_count'] += 1
                lane_measures['lane_count'] += 1
            lane_measures['vehicles'] = vehicles
            full_observation[lane] = lane_measures
        """
        full_observation['num_vehicles'] = all_vehicles
        if self.last_step_vehicles is None:
            full_observation['arrivals'] = full_observation['num_vehicles']
            full_observation['departures'] = set()
        else:
            full_observation['arrivals'] = self.last_step_vehicles.difference(all_vehicles)
            departs = all_vehicles.difference(self.last_step_vehicles)
            full_observation['departures'] = departs
            # Clear departures from waiting times
            for vehicle in departs:
                if vehicle in self.waiting_times: self.waiting_times.pop(vehicle)
        self.last_step_vehicles = all_vehicles
        """
        self.full_observation = full_observation
        self._add_runtime_profile("observe", time.perf_counter() - start)




    def observepart(self, step_length, distance):
        start = time.perf_counter()
        '''
        observe
        Get observation of the whole roadnet, including lane_waiting_time_count, lane_waiting_count, lane_count and queue_length.
        
        :param step_length: time duration of step
        :param distance: distance limitation that it can only get vehicles which are within the length of the road
        :return: None
        '''

        cmd = Registry.mapping['command_mapping']['setting'].param
        penetration_rate = cmd.get('penetration_rate', 1.0)
        use_omnet = cmd.get('use_omnet', False)
        visible_ids = self.get_vehicle_ids_set_csv() if use_omnet else None

        full_observation = dict()
        all_vehicles = set()
        for lane in self.lanes:
            vehicles = []
            lane_measures = {'lane_waiting_time_count': 0, 'lane_waiting_count': 0, 'lane_count': 0, 'queue_length': 0}
            lane_vehicles_all = self._get_vehicles(lane, distance)

            if use_omnet:
                lane_vehicles = [v for v in lane_vehicles_all if v in visible_ids]
            elif lane_vehicles_all:
                num_to_select = max(1, int(len(lane_vehicles_all) * penetration_rate)) if penetration_rate > 0 else 0
                lane_vehicles = random.sample(lane_vehicles_all, min(num_to_select, len(lane_vehicles_all)))
            else:
                lane_vehicles = []

            for v in lane_vehicles:
                all_vehicles.add(v)
                if v in self.waiting_times:
                    self.waiting_times[v] += step_length
                elif self.eng.vehicle.getWaitingTime(v) > 0:
                    self.waiting_times[v] = self.eng.vehicle.getWaitingTime(v)
                v_measures = dict()
                v_measures['name'] = v
                v_measures['wait'] = self.waiting_times[v] if v in self.waiting_times else 0
                #TODO: CHEC ITS RIGHT CALCULATION?
                lane_measures['queue_length'] = lane_measures['queue_length'] + 1
                v_measures['speed'] = self.eng.vehicle.getSpeed(v)
                v_measures['position'] = self.eng.vehicle.getLanePosition(v)
                vehicles.append(v_measures)
                if v_measures['wait'] > 0:
                    lane_measures['lane_waiting_time_count'] += v_measures['wait']
                    lane_measures['lane_waiting_count'] += 1
                lane_measures['lane_count'] += 1
            lane_measures['vehicles'] = vehicles
            full_observation[lane] = lane_measures
        """
        full_observation['num_vehicles'] = all_vehicles
        if self.last_step_vehicles is None:
            full_observation['arrivals'] = full_observation['num_vehicles']
            full_observation['departures'] = set()
        else:
            full_observation['arrivals'] = self.last_step_vehicles.difference(all_vehicles)
            departs = all_vehicles.difference(self.last_step_vehicles)
            full_observation['departures'] = departs
            # Clear departures from waiting times
            for vehicle in departs:
                if vehicle in self.waiting_times: self.waiting_times.pop(vehicle)
        self.last_step_vehicles = all_vehicles
        """
        self.full_observation_part = full_observation
        self._add_runtime_profile("observepart", time.perf_counter() - start)






    def _get_vehicles(self, lane, max_distance):
        '''
        _get_vehicles
        Get number of vehicles running on the specific lane within max distance.
        
        :param lane: lane id
        :param max_distance: distance limitation that it can only get vehicles which are within the length of the lane
        :return detectable: number of vehicles
        '''
        # TODO: reduce complexity -> find all vehicles within max_distance and on this lane
        detectable = []
        for v in self.eng.lane.getLastStepVehicleIDs(lane):
            path = self.eng.vehicle.getNextTLS(v)
            if len(path) > 0:
                next_light = path[0]
                distance = next_light[2]
                if distance <= max_distance:
                    detectable.append(v)
        return detectable



    # TODO: revert x and y
    def _get_direction(self, road, out=True):
        if out:
            x = road[1][0] - road[0][0]
            y = road[1][1] - road[0][1]
        else:
            x = road[-2][0] - road[-1][0]
            y = road[-2][1] - road[-1][1]
        tmp = atan2(x, y)
        return tmp if tmp >= 0 else (tmp + 2 * pi)

    def create_yellows(self, phases, yellow_length, interface_flag):
        # interface_flag: 1:libsumo, 0: traci
        new_phases = copy.copy(phases)
        yellow_dict = {}    # current phase + next phase keyed to corresponding yellow phase index
        # Automatically create yellow phases, traci will report missing phases as it assumes execution by index order
        for i in range(0, len(phases)):
            for j in range(0, len(phases)):
                if i != j:
                    need_yellow, yellow_str = False, ''
                    for sig_idx in range(len(phases[i].state)):
                        if (phases[i].state[sig_idx] == 'G' or phases[i].state[sig_idx] == 'g') and (phases[j].state[sig_idx] == 'r' or phases[j].state[sig_idx] == 's'):
                            need_yellow = True
                            yellow_str += 'r'
                        else:
                            yellow_str += phases[i].state[sig_idx]
                    if need_yellow:  # If a yellow is required
                        if interface_flag:
                            new_phases.append(libsumo.trafficlight.Phase(yellow_length, yellow_str))
                        else:
                            new_phases.append(traci.trafficlight.Phase(yellow_length, yellow_str))
                        yellow_dict[str(i) + '_' + str(j)] = len(new_phases) - 1  # The index of the yellow phase in SUMO
        return new_phases, yellow_dict




@Registry.register_world('sumo')
class World(object):
    '''
    World Class is mainly used for creating a SUMO engine and maintain information about SUMO world.
    '''
    def __init__(self, sumo_config, placeholder=0, **kwargs):
        if kwargs['interface'] == 'libsumo':
            self.interface_flag = True
        elif kwargs['interface'] == 'traci':
            self.interface_flag = False
        else:
            raise Exception('NOT IMPORTED YET')
        cmd = Registry.mapping['command_mapping']['setting'].param
        self.traci_port = int(cmd.get('traci_port', 9999))
        self.traci_connect_retries = int(cmd.get('traci_connect_retries', 120))
        self.traci_connect_delay = float(cmd.get('traci_connect_delay', 1.0))
        self.use_omnet = bool(cmd.get('use_omnet', False))
        self.traci_multi_client = (not self.interface_flag) and self.use_omnet
        with open(sumo_config) as f:
            sumo_dict = json.load(f)
        if sumo_dict['gui'] == "True" or sumo_dict['gui'] == True:
            sumo_cmd = [sumolib.checkBinary('sumo-gui')]
        else:
            sumo_cmd = [sumolib.checkBinary('sumo')]
        client_args = ['--num-clients', '2'] if self.traci_multi_client else []
        if not sumo_dict.get('combined_file'):
            sumo_cmd += ['-n', os.path.join(sumo_dict['dir'], sumo_dict['roadnetFile']),
                         '-r', os.path.join(sumo_dict['dir'], sumo_dict['flowFile']),
                         '--no-warnings', str(sumo_dict['no_warning'])] + client_args
        else:
            sumo_cmd += ['-c', os.path.join(sumo_dict['dir'], sumo_dict['combined_file']),
                         '--no-warnings', str(sumo_dict['no_warning'])] + client_args
        self.net = os.path.join(sumo_dict['dir'], sumo_dict['roadnetFile'])
        self.route = os.path.join(sumo_dict['dir'], sumo_dict['flowFile'])
        self.sumo_cmd = sumo_cmd
        self.warning = sumo_dict['no_warning']
        print("building world...")
        self.connection_name = sumo_dict['name']
        self.map = sumo_dict['roadnetFile'].split('/')[-1].split('.')[0]
        
        if self.interface_flag:
            libsumo.start(sumo_cmd)
            self.eng = libsumo
        else:
            self._start_traci(sumo_cmd, sumo_dict['name'])
        # TODO: roadnet not implemented but not necessary
        self.RIGHT = True  # TODO: currently set to be true
        self.saverr=False
        self.interval = sumo_dict['interval']
        self.step_ratio = 1  # TODO: register in Registry later
        self.step_length = 1  # should be 1 in our setting
        self.max_distance = 200 # TODO: set in registry
        # get all intersections (dict here)
        self.intersection_ids = self.eng.trafficlight.getIDList()
        # prepare phase information for each intersections
        self.green_phases = self.generate_valid_phase()

        # creating all intersections
        self.id2intersection = dict()
        self.intersections = []
        for ts in self.eng.trafficlight.getIDList():
            self.id2intersection[ts] = Intersection(ts, self, self.green_phases[ts])  # this IntSec has different phases
            self.intersections.append(self.id2intersection[ts])
        self.id2idx = {i: idx for idx,i in enumerate(self.id2intersection)}
        # TODO: to see if its necessary to test .intersections or .observe here
        # TODO: to see if pass observation and its shape by generator
        self.all_roads = [x for x in self.eng.edge.getIDList()]
        self.all_lanes = [ x for x in self.eng.lane.getIDList()]
        # for itsec in self.intersections:
        #     for road in itsec.road_lane_mapping.keys():
        #         if itsec.road_lane_mapping[road] and road not in self.all_roads:
        #             # append road name into all_roads if road exists
        #             self.all_roads.append(road)
                    # for lane in itsec.road_lane_mapping[road]:
                    #     if lane not in self.all_lanes:
                    #         self.all_lanes.append(lane)

        # restart eng
        self.run = 0
        self.inside_vehicles = dict()
        self.vehicles = dict()
        for intsec in self.intersections:
            intsec.observe(self.step_length, self.max_distance)
            intsec.observepart(self.step_length, self.max_distance)
        if self.interface_flag:
            if not self.connection_name: 
                libsumo.switch(self.connection_name)  # TODO: make sure what's this step doing
            libsumo.close()
        elif not self.traci_multi_client:
            if not self.connection_name: 
                traci.switch(self.connection_name)  # TODO: make sure what's this step doing
            traci.close()
        else:
            print("[World] Live OMNeT: keeping initial TraCI connection open for inference.")
        # self.connection_name = self.map + '-' + self.connection_name
        if not os.path.exists(os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                           self.connection_name)):
            os.mkdir(os.path.join(Registry.mapping['logger_mapping']['path'].path, self.connection_name))

        print('Connection ID', self.connection_name)

        self.info_functions = {
            "vehicles": self.get_vehicles, # TODO check this func
            "lane_count": self.get_lane_vehicle_count,
            "lane_count_part": self.get_lane_vehicle_count_part,
            "lane_waiting_count": self.get_lane_waiting_vehicle_count,
            "lane_vehicles": self.get_lane_vehicles,
            "time": self.get_current_time,
            "vehicle_distance": None,
            "pressure": self.get_pressure,
            "lane_pressure": self.get_lane_pressure,
            "lane_waiting_time_count": self.get_lane_waiting_time_count,
            "lane_delay": self.get_lane_delay,
            "real_delay": self.get_real_delay,
            "vehicle_trajectory": self.get_vehicle_trajectory,
            "history_vehicles": None,
            "phase": self.get_cur_phase,
            "throughput": self.get_cur_throughput,
            "average_travel_time": None
        }
        self.fns = []
        self.info = {}
        # test generate observation information
        self.vehicle_trajectory = {}
        self.vehicle_maxspeed = {}
        self.real_delay = {}
        self.allcapacity = {}  # capacity division is commented out in pressure calc
        # get in_lanes and out_lanes
        self.in_lanes, self.out_lanes = self.get_in_out_lanes()
        max_speed_file = '/home/exx/Downloads/modified_libsignal-main/max_Speed.json'
        if os.path.exists(max_speed_file):
            with open(max_speed_file) as f:
                self.maxspeeds = json.load(f)
        else:
            self.maxspeeds = {}  # populated dynamically during simulation
        # self.net already set from config above
        self.all_roads = []
        self.edge_nodes = {}
        self.nodes_edge = {}
        self.network_graph = nx.Graph()
        self.build_network()
        self.routes_file = self.route  # derived from config
        self.destinations = self.get_destinations()
        self.reliability, self.criticality_score_lanes = self.initialize_criticality()
        self.oldreliability = self.reliability

    def reset_runtime_profile(self):
        for intsec in getattr(self, "intersections", []):
            intsec.reset_runtime_profile()

    def get_runtime_profile(self):
        totals = {
            "visible_csv_read_count": 0,
            "visible_csv_read_wall_s": 0.0,
            "observe_count": 0,
            "observe_wall_s": 0.0,
            "observepart_count": 0,
            "observepart_wall_s": 0.0,
        }
        for intsec in getattr(self, "intersections", []):
            profile = intsec.get_runtime_profile()
            for key, value in profile.items():
                totals[key] = totals.get(key, 0) + value
        return totals

    def _start_traci(self, sumo_cmd, connection_label):
        """Start SUMO with TraCI and wait until OMNeT (second client) can connect."""
        import time

        last_error = None
        for attempt in range(self.traci_connect_retries):
            try:
                if not connection_label:
                    traci.start(sumo_cmd, port=self.traci_port)
                    traci.setOrder(1)
                    self.eng = traci
                else:
                    traci.start(sumo_cmd, label=connection_label, port=self.traci_port)
                    traci.setOrder(1)
                    self.eng = traci.getConnection(connection_label)
                if attempt > 0:
                    print(
                        f"[World] TraCI connected on port {self.traci_port} "
                        f"after {attempt + 1} attempt(s)."
                    )
                return
            except Exception as exc:
                last_error = exc
                try:
                    traci.close()
                except Exception:
                    pass
                if attempt == 0:
                    print(
                        f"[World] Waiting for OMNeT/TraCI on port {self.traci_port} "
                        f"(up to {self.traci_connect_retries} retries)..."
                    )
                time.sleep(self.traci_connect_delay)
        raise RuntimeError(
            f"Could not start/connect TraCI on port {self.traci_port} after "
            f"{self.traci_connect_retries} attempts. Last error: {last_error}"
        ) from last_error

    def _generate_valid_phase_from_program(self):
        """Read signal phases from SUMO program logic (no simulation steps)."""
        valid_phases = dict()
        for lightID in self.intersection_ids:
            green_phases = []
            seen = set()
            for logic in self.eng.trafficlight.getAllProgramLogics(lightID):
                for phase in logic.phases:
                    state = phase.state
                    if 'y' in state:
                        continue
                    if state.count('r') + state.count('s') == len(state):
                        continue
                    if state in seen:
                        continue
                    seen.add(state)
                    if self.interface_flag:
                        green_phases.append(
                            libsumo.trafficlight.Phase(self.step_length, state)
                        )
                    else:
                        green_phases.append(
                            traci.trafficlight.Phase(self.step_length, state)
                        )
            valid_phases[lightID] = green_phases
        return valid_phases

    def generate_valid_phase(self):
        '''
        generate_valid_phase
        Generate valid phases that will be executed by intersections later.
        
        :param: None
        :return valid_phases: valid phases that will be executed by intersections later.
        '''
        if self.traci_multi_client:
            print(
                "[World] Live OMNeT: reading traffic-light phases from program logic "
                "(no TraCI steps before OMNeT connects)."
            )
            return self._generate_valid_phase_from_program()

        valid_phases = dict()
        for i in range(0, 500):    # TODO grab info. directly from tllogic python interface
            for lightID in self.intersection_ids:
                current_phase = self.eng.trafficlight.getRedYellowGreenState(lightID)
                if not lightID in valid_phases:
                    valid_phases[lightID] = []
                has_phase = False
                for phase in valid_phases[lightID]:
                    if phase == current_phase:
                        has_phase = True
                if not has_phase:
                    valid_phases[lightID].append(current_phase)
            self.step_sim()
        for ts in valid_phases:
            green_phases = []
            for phase in valid_phases[ts]:     # Convert to SUMO phase type
                if 'y' not in phase:
                    if phase.count('r') + phase.count('s') != len(phase):
                        green_phases.append(self.eng.trafficlight.Phase(self.step_length, phase))
            valid_phases[ts] = green_phases
        return valid_phases

    def step_sim(self):
        '''
        step_sim
        Simulate 1s. The monaco scenario expects .25s steps instead of 1s, account for that here.
        
        :param: None
        :return: None
        '''
        # 
        for _ in range(self.step_ratio):
            self.eng.simulationStep()

    def step(self, action=None):
        '''
        step
        Take relative actions and update information.
        
        :param actions: actions list to be executed at all intersections at the next step
        :return: None
        '''
        # TODO: support interval != 1
        if action is not None:
            for i, intersection in enumerate(self.intersections):
                intersection.pseudo_step(action[i])
            self.step_sim()
        for intsec in self.intersections:
            intsec.observe(self.step_length, self.max_distance)
            intsec.observepart(self.step_length, self.max_distance)

        # TODO: register vehicles here
        entering_v = self.eng.simulation.getDepartedIDList()
        for v in entering_v:
            self.inside_vehicles.update({v: self.get_current_time()})
        exiting_v = self.eng.simulation.getArrivedIDList()
        for v in exiting_v:
            self.vehicles.update({v: self.get_current_time() - self.inside_vehicles[v]})
        self._update_infos()
        self.vehicle_trajectory, self.vehicle_maxspeed = self.get_vehicle_trajectory()
        self.run += 1
                # INSERT_YOUR_CODE
        if self.run >= 4000:
            print("[World] Simulation reached 300 seconds, saving results and stopping simulation.")
            # Save core SUMO metrics results before exiting
            _att_fn = self.info_functions.get("average_travel_time")
            results = {
                "throughput": self.get_cur_throughput(),

                "average_travel_time": _att_fn() if callable(_att_fn) else None,
                "lane_delay": self.get_lane_delay(),
                "real_delay": self.get_real_delay()
            }
            with open("/home/exx/Desktop/vtc2026/LibSignal-master/experiments/shortsimulationsresults/sumo_metrics_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"[World] Saved sumo_metrics_results.json with keys {list(results.keys())}")
            raise SystemExit
        #if (self.run-1)%1==0:
        #    self.criticalities,self.reliability,self.criticality_score_lanes=self.percolation()#self.edge_betweenness_centrality()#self.percolation()
        #    if self.saverr==True:
        #        self.savereliability(self.run,self.reliability)


    def get_links_average_speed(self):
        allspeedsonlane = dict()
        for intsec in self.intersections:
            obs = intsec.full_observation_part or intsec.full_observation or {}
            for lane in intsec.lanes:
                allspeedsonlane.setdefault(lane, [])
                if lane not in obs:
                    continue
                vehicles = obs[lane]['vehicles']
                if lane not in allspeedsonlane:
                    allspeedsonlane[lane] = []
                #print(vehicles)  # Debugging
                for v in vehicles:
                    allspeedsonlane[lane].append(v['speed'])
        #print(list(allspeedsonlane.keys()))  # Debugging
        return allspeedsonlane

    def get_qualities(self):
        lanes_dic = self.get_links_average_speed()

        # Calculate average speed for each lane directly
        result = {key: (sum(value) / len(value)) if value else 1 for key, value in lanes_dic.items()}

        qualities = {}
        edge_qualities = {}

        # Efficiently calculate the qualities based on maxspeeds
        for lane, avg_speed in result.items():
            # Update maxspeeds if avg_speed is higher or lane is not in maxspeeds
            if lane not in self.maxspeeds or avg_speed > self.maxspeeds[lane]:
                self.maxspeeds[lane] = avg_speed

            # Normalize quality, handling the case where max speed is 0
            qualities[lane] = avg_speed / self.maxspeeds[lane] if self.maxspeeds[lane] > 0 else 1

            # Extract the edge ID from the lane (assuming the edge ID is everything before the last underscore)
            edge_id = '_'.join(lane.split('_')[:-1])

            # Accumulate qualities for edges
            if edge_id not in edge_qualities:
                edge_qualities[edge_id] = []
            edge_qualities[edge_id].append(qualities[lane])

        # Calculate average quality for each edge
        edge_qualities = {edge: sum(lane_qualities) / len(lane_qualities) for edge, lane_qualities in edge_qualities.items()}

        #print('qualities:', qualities)
        #print('edge_qualities:', edge_qualities)

        return  edge_qualities


    def get_od(self):
        """
        Get the origin-destination (OD) count for all vehicles.
        The origin is the current position of the vehicle (either an intersection or the origin node of a lane),
        and the destination is taken from self.destinations.
        Returns a dictionary with (origin, destination) as the key and the count of occurrences as the value.
        """
        od_count = {}
        active_vehicles = set(self.eng.vehicle.getIDList())  # Get the list of currently active vehicles
        
        # Loop through each vehicle in the vehicle_trajectory dictionary
        for vehicle_id, trajectory in self.vehicle_trajectory.items():
            #print(vehicle_id)
            #print(trajectory)
            if vehicle_id in active_vehicles:  # Check if the vehicle is still in the network
                
                if trajectory:
                    # Get the last recorded lane (most recent position) for the vehicle
                    current_lane = trajectory[-1][0]  # The first element in the sublist is the lane ID

                    # Extract the edge associated with the lane (remove the last underscore and number)
                    edge = '_'.join(current_lane.split('_')[:-1])

                    # Determine the origin (either an intersection or the origin of the lane)
                    if edge in self.edge_nodes:
                        # It's a lane, map it to the origin intersection (from_node of the edge)
                        origin_intersection = self.edge_nodes[edge][0]
                        origin = origin_intersection
                    else:
                        # It's an intersection, use it as the origin
                        origin = current_lane

                    # Get the destination from self.destinations for this vehicle
                    destination = self.destinations.get(vehicle_id, None)
                    #print(destination)
                    # If both origin and destination are available, count the OD pair
                    if origin and destination:
                        od_pair = (origin, destination)
                        if od_pair not in od_count:
                            od_count[od_pair] = 0
                        od_count[od_pair] += 1  # Increment count for this OD pair

        # Return the OD count dictionary
        return od_count

    def build_network(self):
        """
        Build a NetworkX graph by extracting edges and their corresponding nodes (junctions)
        from the SUMO road network, and assign quality values to the edges based on actual data.
        """
        net = sumolib.net.readNet(self.net)  # Read the SUMO network file

        # Extract all edges and their corresponding from/to nodes (junctions)
        for edge in net.getEdges():
            edge_id = edge.getID()
            if edge_id.startswith(':'):
                continue

            from_node = edge.getFromNode().getID()  # Get the starting node of the edge
            to_node = edge.getToNode().getID()      # Get the ending node of the edge

            # Add edge information to the list of all roads and store the nodes associated with the edge
            self.all_roads.append(edge_id)
            self.edge_nodes[edge_id] = (from_node, to_node)
            self.nodes_edge[(from_node, to_node)] = edge_id

            # Get the quality of this edge (use the actual get_quality method)
            quality = self.get_qualities().get(edge_id, 1)

            # Add the edge to the NetworkX graph with the quality as an edge attribute
            self.network_graph.add_edge(from_node, to_node, q=quality)

        # Collapse or remove internal SUMO nodes (e.g., nodes starting with ':')
        nodes_to_remove = [node for node in self.network_graph.nodes if node.startswith(':')]
        for node in nodes_to_remove:
            connected_nodes = list(self.network_graph.adj[node])

            # If the node is connected to only one other node, skip it (it's redundant)
            if len(connected_nodes) == 1:
                self.network_graph.remove_node(node)
                continue

            # Collapse connections: connect its neighbors directly to each other
            for i in range(len(connected_nodes)):
                for j in range(i + 1, len(connected_nodes)):
                    u, v = connected_nodes[i], connected_nodes[j]
                    # Avoid duplicate edges
                    if not self.network_graph.has_edge(u, v):
                        self.network_graph.add_edge(u, v, q=self.network_graph[node][u]['q'])

            # Remove the internal node
            self.network_graph.remove_node(node)

        print("NetworkX graph built with {} edges after cleaning internal nodes.".format(len(self.network_graph.edges)))

    def get_destinations(self):
        """
        Extract vehicle destinations from the routes file.
        Returns a dictionary with vehicle ID as the key and the destination edge as the value.
        """
        vehicle_destinations = {}

        # Parse the XML routes file to extract vehicle destinations
        tree = ET.parse(self.routes_file)
        root = tree.getroot()

        # Loop through each trip or flow in the routes file
        for trip in root.findall('trip'):
            vehicle_id = trip.get('id')  # Get the trip ID
            destination_edge = trip.get('to')  # Get the destination edge (the "to" attribute)

            if destination_edge:
                vehicle_destinations[vehicle_id] = self.edge_nodes[destination_edge][1]   # Store the destination edge

        for flow in root.findall('flow'):
            flow_id = flow.get('id')  # Get the flow ID
            destination_edge = flow.get('to')  # Get the destination edge (the "to" attribute)

            if destination_edge:
                vehicle_destinations[flow_id] = self.edge_nodes[destination_edge][1] #destination_edge  # Store the destination edge
                #print(self.edge_nodes[destination_edge])

        return vehicle_destinations


    def percolation(self):
        """
        Perform percolation analysis on the current NetworkX graph using OD pairs from the SUMO world.
        Returns criticality scores for each edge and the network reliability (alpha).
        """
        def maximum_capacity_paths(G, source, weight, target=None):
            """
            Find the limiting link (edge) between a source and all targets on the graph G.
            This uses a modified Dijkstra's algorithm to find paths based on edge capacity (quality).
            """
            get_weight = lambda u, v, data: data.get(weight, 1)
            paths = {source: [source]}  # Store paths
            G_succ = G.succ if G.is_directed() else G.adj

            dist = {}  # Final distances
            pred = {}  # Predecessor nodes
            seen = {source: 0}
            fringe = []
            heappush(fringe, (-1, next(count()), source))

            while fringe:
                (d, _, v) = heappop(fringe)
                if v in dist:
                    #print('continue1')
                    continue  # Node already processed
                dist[v] = d
                if v == target:
                    #print('break')
                    break
                for u, e in G_succ[v].items():
                    if u == source:
                        #print('continue2')
                        continue
                    vu_dist = max([dist[v], -get_weight(v, u, e)])
                    if u not in seen or vu_dist < seen[u]:
                        seen[u] = vu_dist
                        heappush(fringe, (vu_dist, next(count()), u))
                        paths[u] = paths[v] + [u]
                        pred[u] = (v, u)

            dist = {i: -dist[i] for i in dist}
            return dist, pred

        # Get the network graph
        G = self.network_graph
        #print("Nodes in the graph:", list(G.nodes))
        # Get the OD pairs using the existing get_od() function
        OD_pairs = self.get_od()
        #print(OD_pairs)
        total_demand = sum([1 for _ in OD_pairs])  # Total number of OD pairs (assuming unit demand per vehicle)

        criticality_scores = {}
        
        # Loop over each origin node in the network
        for origin in G.nodes:
            #print('OP')
            # Find limiting links between origin and all other nodes
            _, limiting_links_dict = maximum_capacity_paths(G, origin, weight='q')
            #print(limiting_links_dict)
            #print('OP2')
            for destination in limiting_links_dict:
                #print('origin:')
                #print(origin)
                #print('destination:')
                #print(destination)
                #print('od pairs')
                #print(OD_pairs)
                if origin != destination and (origin, destination) in OD_pairs:

                    # Update criticality score for each limiting link
                    limiting_link = limiting_links_dict[destination]
                    if limiting_link not in criticality_scores:
                        criticality_scores[limiting_link] = 0
                    criticality_scores[limiting_link] += 1 / float(total_demand)

        # Calculate reliability (alpha)
        alpha = sum([G[u][v]['q'] * criticality_scores.get((u, v), 0) for u, v in G.edges])
        criticality_score_edges={self.nodes_edge[k]:criticality_scores[k] for k in criticality_scores.keys()}
        #print(f"Criticalities: {criticality_score_edges}")
        for road in self.all_roads:
            if road not in criticality_score_edges.keys():
                criticality_score_edges[road]=0
        #print(len(criticality_score_edges))
        #print(self.all_roads)
        #print(f"Reliability (Alpha): {alpha}")
        criticality_score_lanes = {}

        # Loop over all the lanes in the SUMO network
        for lane_id in self.all_lanes:
            # Extract the edge_id from the lane_id (edge_id is everything before the last underscore)
            edge_id = '_'.join(lane_id.split('_')[:-1])

            # Assign the criticality score of the edge to the lane
            if edge_id in criticality_score_edges:
                criticality_score_lanes[lane_id] = criticality_score_edges[edge_id]
            else:
                #print('ELSEEE')
                # If the edge has no criticality score, default to 0
                criticality_score_lanes[lane_id] = 0
        #print(criticality_score_lanes)
        
        return criticality_scores, alpha,criticality_score_lanes




    def initialize_criticality(self):
        criticality_score_lanes={}
        reliability={}
        for lane_id in self.all_lanes:
        # Extract the edge_id from the lane_id (edge_id is everything before the last underscore)
        #edge_id = '_'.join(lane_id.split('_')[:-1])
            # Assign the criticality score of the edge to the lane
            criticality_score_lanes[lane_id] = 0
            reliability[lane_id] = 0

        return 0,criticality_score_lanes



    def savereliability(self, run,reliability, file_path="reliability_data.csv"):
        """
        Saves the run and reliability information to a CSV file.
        
        :param run: Current run iteration (int).
        :param reliability: The reliability score or data (float or list).
        :param file_path: The file path where the data should be saved (default is "reliability_data.csv").
        :return: None
        """


        data_to_save = {'Run': [run], 'Reliability': [reliability]}

        # Check if the file already exists
        if os.path.exists(file_path):
            # Load the existing file and append new data
            existing_data = pd.read_csv(file_path)
            new_data = pd.DataFrame(data_to_save)
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            combined_data.to_csv(file_path, index=False)
        else:
            # Create a new file with the initial data
            initial_data = pd.DataFrame(data_to_save)
            initial_data.to_csv(file_path, index=False)



    def reset(self):
        '''
        reset
        reset information, including vehicles, vehicle_trajectory, etc.
       
        :param: None
        :return: None
        '''
        had_active_run = self.run != 0
        if had_active_run:
            # TODO: test why need switch in original code
            if self.interface_flag:
                libsumo.close()
            else:
                traci.close()
        self.run = 0
        self.vehicles = dict()
        self.inside_vehicles = dict()
        # TODO: check when to close traci
        if self.interface_flag:
            libsumo.start(self.sumo_cmd)
            # TODO: set trip info output
            self.eng = libsumo
        elif self.traci_multi_client and not had_active_run:
            print("[World] Live OMNeT: reusing initial TraCI connection on reset.")
        else:
            self._start_traci(self.sumo_cmd, self.connection_name)
        self.id2intersection = dict()
        self.intersections = []
        for ts in self.eng.trafficlight.getIDList():
            self.id2intersection[ts] = Intersection(ts, self, self.green_phases[ts])  # this IntSec has different phases
            self.intersections.append(self.id2intersection[ts])
        self.id2idx = {i: idx for idx,i in enumerate(self.id2intersection)}

        for intsec in self.intersections:
            intsec.observe(self.step_length, self.max_distance)
            intsec.observepart(self.step_length, self.max_distance)

        self._update_infos()
        # TODO: check if its the problem
        entering_v = self.eng.simulation.getDepartedIDList()
        for v in entering_v:
            self.inside_vehicles.update({v: self.get_current_time()})
        self.vehicle_trajectory = {}
        self.vehicle_maxspeed = {}
        self.real_delay= {}

    def get_current_time(self):
        '''
        get_current_time
        Get simulation time (in seconds).
        
        :param: None
        :return result: current time
        '''
        result = self.eng.simulation.getTime()
        return result

    def get_vehicles(self):
        '''
        get_vehicles
        Get all vehicle ids.
        
        :param: None
        :return: None
        '''
        result = 0
        count = 0
        for v in self.vehicles.keys():
            count += 1
            result += self.vehicles[v]
        if count == 0:
            return 0
        else:
            return result/count

    def subscribe(self, fns):
        '''
        subscribe
        Subscribe information you want to get when training the model.
        
        :param fns: information name you want to get
        :return: None
        '''
        if isinstance(fns, str):
            fns = [fns]
        for fn in fns:
            if fn in self.info_functions:
                if fn not in self.fns:
                    self.fns.append(fn)
            else:
                raise Exception(f'Info function {fn} not implemented')

    def get_info(self, info):
        '''
        get_info
        Get specific information.
        
        :param info: the name of the specific information
        :return _info: specific information
        '''
        _info = self.info[info]
        return _info

    def _update_infos(self):
        '''
        _update_infos
        Update global information after reset or each step.
        
        :param: None
        :return: None
        '''
        self.info = {}
        for fn in self.fns:
            self.info[fn] = self.info_functions[fn]()

    def get_lane_vehicle_count(self):
        '''
        get_lane_vehicle_count
        Get number of vehicles in each lane.
        
        :param: None
        :return result: number of vehicles in each lane
        '''
        result = dict()
        for intsec in self.intersections:
            obs = intsec.full_observation_part or intsec.full_observation or {}
            for lane in intsec.lanes:
                result[lane] = obs[lane]['lane_count'] if lane in obs else 0
        return result

    def get_lane_vehicle_count_part(self):
        '''
        get_lane_vehicle_count
        Get number of vehicles in each lane (partial observation).
        
        :param: None
        :return result: number of vehicles in each lane
        '''
        result = dict()
        for intsec in self.intersections:
            obs = intsec.full_observation_part or intsec.full_observation or {}
            for lane in intsec.lanes:
                result[lane] = obs[lane]['lane_count'] if lane in obs else 0
        return result

    def get_pressure(self):
        '''
        get_pressure
        Get pressure of each intersection. 
        Pressure of an intersection equals to number of vehicles that in in_lanes minus number of vehicles that in out_lanes.
        
        :param: None
        :return pressures: pressure of each intersection
        '''
        pressures = dict()
        lane_vehicles = self.get_lane_vehicle_count_part()
        for i in self.intersections:
            pressure = 0
            for road in i.in_roads:
                for k in i.road_lane_mapping[road]:
                    pressure += lane_vehicles[k]#/ self.allcapacity[k]
            for road in i.out_roads:
                for k in i.road_lane_mapping[road]:
                    pressure -= lane_vehicles[k]#/ self.allcapacity[k]
            pressures[i.id] = pressure#*100
        return pressures
        
    def get_in_out_lanes(self):
        in_lanes = []
        out_lanes = []
        for i in self.intersections:
            for road in i.in_roads:
                for lane in i.road_lane_mapping[road]:
                    in_lanes.append(lane)
            for road in i.out_roads:
                for lane in i.road_lane_mapping[road]:
                    out_lanes.append(lane)
        # add in_lanes of virtual intersections which can be regarded as out_lanes of non-virtual intersections.
        for lane in self.all_lanes:
            if lane not in out_lanes:
                out_lanes.append(lane)
        return in_lanes, out_lanes

    def get_lane_pressure(self):
        '''
        get_lane_pressure
        Get pressure of each lane in an intersection. 
        Pressure of each lane equals to number of vehicles that in the in_lane minus number of vehicles that in out_lane.
        
        :param: None
        :return pressures: pressure of each lane
        '''
        lvc = self.get_lane_vehicle_count()
        pressures = {}
        pressures = {x:0 for x in self.in_lanes}
        for inter_obj in self.intersections:
            for lanelink in inter_obj.lanelinks:
                start, end = lanelink[0][0], lanelink[0][1]
                pressures[start] += lvc[start]
                pressures[start] -= lvc[end]
        return pressures

    def get_lane_waiting_time_count(self):
        '''
        get_lane_waiting_time_count
        Get waiting time of vehicles in each lane.
        
        :param: None
        :return result: waiting time of vehicles in each lane
        '''
        result = dict()
        for intsec in self.intersections:
            obs = intsec.full_observation_part or intsec.full_observation or {}
            for lane in intsec.lanes:
                result[lane] = obs[lane]['lane_waiting_time_count'] if lane in obs else 0
        return result

    def get_lane_waiting_vehicle_count(self):
        '''
        get_lane_waiting_vehicle_count
        Get number of waiting vehicles in each lane.
        
        :param: None
        :return result: number of waiting vehicles in each lane
        '''
        result = dict()
        for intsec in self.intersections:
            obs = intsec.full_observation_part or intsec.full_observation or {}
            for lane in intsec.lanes:
                result[lane] = obs[lane]['lane_waiting_count'] if lane in obs else 0
        return result

    def get_cur_phase(self):
        '''
        get_cur_phase
        Get current phase of each intersection.

        :param: None
        :return result: current phase of each intersection
        '''
        result = []
        for intsec in self.intersections:
            result.append(intsec.get_current_phase())
        return result

    def get_average_travel_time(self):
        '''
        get_average_travel_time
        Get average travel time of all vehicles.
        
        :param: None
        :return tvg_time: average travel time of all vehicles
        '''
        tvg_time = self.get_vehicles()
        return tvg_time

    def get_lane_vehicles(self):
        '''
        get_lane_vehicles
        Get vehicles' id of each lane.

        :param: None
        :return vehicle_lane: vehicles' id of each lane
        '''
        result = dict()
        for inter in self.intersections:
            obs = inter.full_observation_part or inter.full_observation or {}
            for key in obs.keys():
                result[key] = obs[key]
        return result

    def get_lane_queue_length(self):
        '''
        get_lane_queue_length
        Get queue length of all lanes in the traffic network.
        
        :param: None
        :return result: queue length of all lanes
        '''
        #TODO: CHECK DEFINATION
        result = dict()
        for inter in self.intersections:
            obs = inter.full_observation_part or inter.full_observation or {}
            for key in obs.keys():
                result[key] = obs[key]['queue_length']
        return result

    def get_lane_delay(self):
        '''
        get_lane_delay
        Get approximate delay of each lane. 
        Approximate delay of each lane equals to (1 - lane_avg_speed)/lane_speed_limit.
        
        :param: None
        :return lane_delay: approximate delay of each lane
        '''
        # the delay of each lane: 1 - lane_avg_speed/speed_limit
        # set speed limit to 11.11 by default
        lane_vehicles = self.get_lane_vehicles()
        lane_delay = dict()
        for key in lane_vehicles.keys():
            vehicles = lane_vehicles[key]['vehicles']
            lane_vehicle_count = len(vehicles)
            lane_avg_speed = 0.0
            speed_limit = self.eng.lane.getMaxSpeed(key)
            for vehicle in vehicles:
                speed = vehicle['speed']
                lane_avg_speed += speed
            if lane_vehicle_count == 0:
                lane_avg_speed = speed_limit
            else:
                lane_avg_speed /= lane_vehicle_count
            lane_delay[key] = 1 - lane_avg_speed / speed_limit
        return lane_delay

    # def get_plan_depart_time(self):
    #     """
    #     Get planned depart time for all vehicles appeared in sumo.rou.xml file.
    #     In SUMO and Cityflow, travel time = arriving time-planned depart time.
    #     Note: Not real depart time, but planned depart time.
    #     return: planned depart time of all vehicles.
    #     """
    #     vehicles_all = dict()
    #     tree = ET.parse(self.route)
    #     root = tree.getroot()
    #     vehicles_all.update({obj.attrib['id']: int(float(obj.attrib['depart'])) \
    #         for obj in root.iter('vehicle')})
    #     return vehicles_all

    def get_cur_throughput(self):
        '''
        get_cur_throughput
        Get vehicles' count in the whole roadnet at current step.

        :param: None
        :return throughput: throughput in the whole roadnet at current step
        '''
        throughput = len(self.vehicles)
        # TODO: check if only trach left cars
        return throughput

    def get_vehicle_lane(self):
        '''
        get_vehicle_lane
        Get current lane id and max speed of each vehicle that is running.

        :param: None
        :return vehicle_lane: current lane id of each vehicle
        :return vehicle_maxspeed: max speed of each vehicle
        '''
        # get the current lane of each vehicle. {vehicle_id: lane_id}
        vehicle_lane = {}
        for lane in self.all_lanes:
            vehicles = 	self.eng.lane.getLastStepVehicleIDs(lane)
            for vehicle in vehicles:
                vehicle_lane[vehicle] = lane
                self.vehicle_maxspeed[(vehicle,lane)] = self.eng.vehicle.getAllowedSpeed(vehicle)
        return vehicle_lane, self.vehicle_maxspeed

    def get_vehicle_trajectory(self):
        '''
        get_vehicle_trajectory
        Get trajectory of vehicles that have entered in roadnet, including vehicle_id, enter time, leave time or current time.
        
        :param: None
        :return vehicle_trajectory: trajectory of vehicles that have entered in roadnet
        :return vehicle_maxspeed: max speed of each vehicle that have entered in roadnet
        '''
        # lane_id and time spent on the corresponding lane that each vehicle went through
        vehicle_lane, self.vehicle_maxspeed = self.get_vehicle_lane() # get vehicles on tne roads except turning
        vehicles = list(self.eng.vehicle.getIDList())
        # vehicles = [x for x in vehicle_lane]
        for vehicle in vehicles:
            if vehicle not in self.vehicle_trajectory:
                self.vehicle_trajectory[vehicle] = [[vehicle_lane[vehicle], int(self.eng.simulation.getTime()), 0]]
            else:
                if vehicle not in vehicle_lane.keys(): # vehicle is turning
                    continue
                if vehicle_lane[vehicle] == self.vehicle_trajectory[vehicle][-1][0]: # vehicle is running on the same lane 
                    self.vehicle_trajectory[vehicle][-1][2] += 1
                else: # vehicle has changed the lane
                    self.vehicle_trajectory[vehicle].append(
                        [vehicle_lane[vehicle], int(self.eng.simulation.getTime()), 0])
        return self.vehicle_trajectory, self.vehicle_maxspeed

    def get_real_delay(self):
        '''
        get_real_delay
        Calculate average real delay. 
        Real delay of a vehicle is defined as the time a vehicle has traveled within the environment minus the expected travel time.
        
        :param: None
        :return avg_delay: average real delay of all vehicles
        '''
        self.vehicle_trajectory, self.vehicle_maxspeed = self.get_vehicle_trajectory()
        for v in self.vehicle_trajectory:
            # get road level routes of vehicle
            routes = self.vehicle_trajectory[v] # lane_level
            for idx, lane in enumerate(routes):
                speed = min(self.eng.lane.getMaxSpeed(lane[0]), self.vehicle_maxspeed[(v,lane[0])])
                lane_length = self.eng.lane.getLength(lane[0])
                if idx == len(routes)-1: # the last lane
                    # judge whether the vehicle run over the whole lane.
                    lane_length = self.eng.vehicle.getLanePosition(v) if v in self.eng.vehicle.getIDList() else lane_length
                planned_tt = float(lane_length)/speed
                real_delay = lane[-1] - planned_tt if lane[-1]>planned_tt else 0.
                if v not in self.real_delay.keys():
                    self.real_delay[v] = real_delay
                else:
                    self.real_delay[v] += real_delay

        avg_delay = 0.
        count = 0
        for dic in self.real_delay.items():
            avg_delay += dic[1]
            count += 1
        avg_delay = avg_delay / count
        return avg_delay



