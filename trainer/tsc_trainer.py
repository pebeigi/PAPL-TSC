import os
import time
import csv
import numpy as np
from common.metrics import Metrics
from environment import TSCEnv
from common.registry import Registry
from trainer.base_trainer import BaseTrainer


@Registry.register_trainer("tsc")
class TSCTrainer(BaseTrainer):
    '''
    Register TSCTrainer for traffic signal control tasks.
    '''
    def __init__(
        self,
        logger,
        gpu=0,
        cpu=False,
        name="tsc"
    ):
        super().__init__(
            logger=logger,
            gpu=gpu,
            cpu=cpu,
            name=name
        )
        self.episodes = Registry.mapping['trainer_mapping']['setting'].param['episodes']
        self.steps = Registry.mapping['trainer_mapping']['setting'].param['steps']
        self.test_steps = Registry.mapping['trainer_mapping']['setting'].param['test_steps']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.action_interval = Registry.mapping['trainer_mapping']['setting'].param['action_interval']
        self.save_rate = Registry.mapping['logger_mapping']['setting'].param['save_rate']
        self.learning_start = Registry.mapping['trainer_mapping']['setting'].param['learning_start']
        self.update_model_rate = Registry.mapping['trainer_mapping']['setting'].param['update_model_rate']
        self.update_target_rate = Registry.mapping['trainer_mapping']['setting'].param['update_target_rate']
        self.test_when_train = Registry.mapping['trainer_mapping']['setting'].param['test_when_train']
        # replay file is only valid in cityflow now. 
        # TODO: support SUMO and Openengine later
        
        # TODO: support other dataset in the future
        self.dataset = Registry.mapping['dataset_mapping'][Registry.mapping['command_mapping']['setting'].param['dataset']](
            os.path.join(Registry.mapping['logger_mapping']['path'].path,
                         Registry.mapping['logger_mapping']['setting'].param['data_dir'])
        )
        self.dataset.initiate(ep=self.episodes, step=self.steps, interval=self.action_interval)
        self.yellow_time = Registry.mapping['trainer_mapping']['setting'].param['yellow_length']
        # consists of path of output dir + log_dir + file handlers name
        self.log_file = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                     Registry.mapping['logger_mapping']['setting'].param['log_dir'],
                                     os.path.basename(self.logger.handlers[-1].baseFilename).rstrip('_BRF.log') + '_DTL.log'
                                     )
        self.runtime_file = self.log_file.replace('_DTL.log', '_RUNTIME.csv')
        self.progress_interval = 10

    def create_world(self):
        '''
        create_world
        Create world, currently support CityFlow World, SUMO World and Citypb World.

        :param: None
        :return: None
        '''
        # traffic setting is in the world mapping
        self.world = Registry.mapping['world_mapping'][Registry.mapping['command_mapping']['setting'].param['world']](
            self.path, Registry.mapping['command_mapping']['setting'].param['thread_num'],interface=Registry.mapping['command_mapping']['setting'].param['interface'])

    def create_metrics(self):
        '''
        create_metrics
        Create metrics to evaluate model performance, currently support reward, queue length, delay(approximate or real) and throughput.

        :param: None
        :return: None
        '''
        if Registry.mapping['command_mapping']['setting'].param['delay_type'] == 'apx':
            lane_metrics = ['rewards', 'queue', 'delay']
            world_metrics = ['real avg travel time', 'throughput']
        else:
            lane_metrics = ['rewards', 'queue']
            world_metrics = ['delay', 'real avg travel time', 'throughput']
        self.metric = Metrics(lane_metrics, world_metrics, self.world, self.agents)

    def create_agents(self):
        '''
        create_agents
        Create agents for traffic signal control tasks.

        :param: None
        :return: None
        '''
        self.agents = []
        agent = Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](self.world, 0)
        num_agent = int(len(self.world.intersections) / agent.sub_agents)
        self.agents.append(agent)  # initialized N agents for traffic light control
        for i in range(1, num_agent):
            self.agents.append(Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](self.world, i))

        # for magd agents should share information 
        if Registry.mapping['model_mapping']['setting'].param['name'] == 'magd':
            for ag in self.agents:
                ag.link_agents(self.agents)

    def create_env(self):
        '''
        create_env
        Create simulation environment for communication with agents.

        :param: None
        :return: None
        '''
        # TODO: finalized list or non list
        self.env = TSCEnv(self.world, self.agents, self.metric)

    def _safe_call(self, func, default=None):
        try:
            return func()
        except Exception:
            return default

    def _runtime_count(self, profile, key):
        return int(profile.get(f"{key}_count", 0) or 0)

    def _runtime_wall(self, profile, key):
        return float(profile.get(f"{key}_wall_s", 0.0) or 0.0)

    def _runtime_mean_ms(self, profile, key):
        count = self._runtime_count(profile, key)
        if count <= 0:
            return 0.0
        return 1000.0 * self._runtime_wall(profile, key) / count

    def _write_runtime_profile(self, row):
        os.makedirs(os.path.dirname(self.runtime_file), exist_ok=True)
        fieldnames = [
            "model",
            "mode",
            "use_omnet",
            "interface",
            "simulated_horizon_s",
            "action_interval_s",
            "decisions",
            "reset_wall_s",
            "loop_wall_s",
            "total_wall_s",
            "wall_per_sim_s",
            "real_time_factor",
            "decision_wall_mean_s",
            "action_select_total_s",
            "action_select_mean_ms",
            "env_step_total_s",
            "env_step_mean_ms",
            "visible_csv_read_count",
            "visible_csv_read_total_s",
            "visible_csv_read_mean_ms",
            "observe_count",
            "observe_total_s",
            "observe_mean_ms",
            "observepart_count",
            "observepart_total_s",
            "observepart_mean_ms",
        ]
        write_header = not os.path.exists(self.runtime_file)
        with open(self.runtime_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _log_runtime_profile(
        self,
        mode,
        simulated_horizon_s,
        decisions,
        reset_wall_s,
        loop_wall_s,
        total_wall_s,
        action_select_total_s,
        env_step_total_s,
    ):
        cmd = Registry.mapping['command_mapping']['setting'].param
        world_profile = self._safe_call(self.world.get_runtime_profile, {}) or {}
        wall_per_sim_s = total_wall_s / simulated_horizon_s if simulated_horizon_s else 0.0
        real_time_factor = simulated_horizon_s / total_wall_s if total_wall_s else 0.0
        row = {
            "model": Registry.mapping['model_mapping']['setting'].param['name'],
            "mode": mode,
            "use_omnet": bool(cmd.get("use_omnet", False)),
            "interface": cmd.get("interface"),
            "simulated_horizon_s": simulated_horizon_s,
            "action_interval_s": self.action_interval,
            "decisions": decisions,
            "reset_wall_s": reset_wall_s,
            "loop_wall_s": loop_wall_s,
            "total_wall_s": total_wall_s,
            "wall_per_sim_s": wall_per_sim_s,
            "real_time_factor": real_time_factor,
            "decision_wall_mean_s": loop_wall_s / decisions if decisions else 0.0,
            "action_select_total_s": action_select_total_s,
            "action_select_mean_ms": 1000.0 * action_select_total_s / decisions if decisions else 0.0,
            "env_step_total_s": env_step_total_s,
            "env_step_mean_ms": 1000.0 * env_step_total_s / simulated_horizon_s if simulated_horizon_s else 0.0,
            "visible_csv_read_count": self._runtime_count(world_profile, "visible_csv_read"),
            "visible_csv_read_total_s": self._runtime_wall(world_profile, "visible_csv_read"),
            "visible_csv_read_mean_ms": self._runtime_mean_ms(world_profile, "visible_csv_read"),
            "observe_count": self._runtime_count(world_profile, "observe"),
            "observe_total_s": self._runtime_wall(world_profile, "observe"),
            "observe_mean_ms": self._runtime_mean_ms(world_profile, "observe"),
            "observepart_count": self._runtime_count(world_profile, "observepart"),
            "observepart_total_s": self._runtime_wall(world_profile, "observepart"),
            "observepart_mean_ms": self._runtime_mean_ms(world_profile, "observepart"),
        }
        self._write_runtime_profile(row)
        print(
            "[RUNTIME {mode}] total_wall={total:.2f}s simulated={sim:.1f}s "
            "real_time_factor={rtf:.3f}x action_mean={action:.3f}ms "
            "env_step_mean={step:.3f}ms csv_read_mean={csv_read:.3f}ms".format(
                mode=mode,
                total=total_wall_s,
                sim=simulated_horizon_s,
                rtf=real_time_factor,
                action=row["action_select_mean_ms"],
                step=row["env_step_mean_ms"],
                csv_read=row["visible_csv_read_mean_ms"],
            ),
            flush=True,
        )
        self.logger.info(f"Runtime profile written to {self.runtime_file}: {row}")

    def _live_status(self, mode, step, total_steps, wall_start):
        world = getattr(self, "world", None)
        if world is None:
            return

        sim_time = self._safe_call(world.get_current_time, "?")
        throughput = self._safe_call(world.get_cur_throughput, 0)
        active = self._safe_call(lambda: world.eng.vehicle.getIDCount(), 0)
        departed = len(getattr(world, "inside_vehicles", {}))
        arrived = len(getattr(world, "vehicles", {}))
        visible = "-"
        cmd = Registry.mapping['command_mapping']['setting'].param
        if cmd.get("use_omnet", False) and hasattr(world, "get_vehicle_ids_set_csv"):
            visible = len(self._safe_call(world.get_vehicle_ids_set_csv, set()))

        print(
            "[SUMO {mode}] sim_time={sim_time}s step={step}/{total_steps} "
            "active={active} departed={departed} arrived={arrived} "
            "throughput={throughput} visible_omnet={visible} "
            "queue={queue:.2f} delay={delay:.2f} reward={reward:.2f} "
            "wall={wall:.1f}s".format(
                mode=mode,
                sim_time=sim_time,
                step=step,
                total_steps=total_steps,
                active=active,
                departed=departed,
                arrived=arrived,
                throughput=throughput,
                visible=visible,
                queue=self.metric.queue(),
                delay=self.metric.delay(),
                reward=self.metric.rewards(),
                wall=time.time() - wall_start,
            ),
            flush=True,
        )

    def train(self):
        '''
        train
        Train the agent(s).

        :param: None
        :return: None
        '''
        total_decision_num = 0
        flush = 0
        for e in range(self.episodes):
            if e>=self.episodes-3:
                self.world.saverr=True
            else:
                self.world.saverr=False
            # TODO: check this reset agent
            self.metric.clear()
            last_obs = self.env.reset()  # agent * [sub_agent, feature]

            for a in self.agents:
                a.reset()
            if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
                if self.save_replay and e % self.save_rate == 0:
                    self.env.eng.set_save_replay(True)
                    self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"episode_{e}.txt"))
                else:
                    self.env.eng.set_save_replay(False)
            episode_loss = []
            i = 0
            while i < self.steps:
                if i % self.action_interval == 0:
                    last_phase = np.stack([ag.get_phase() for ag in self.agents])  # [agent, intersections]

                    if total_decision_num > self.learning_start:
                        actions = []
                        for idx, ag in enumerate(self.agents):
                            actions.append(ag.get_action(last_obs[idx], last_phase[idx], test=False))                            
                        actions = np.stack(actions)  # [agent, intersections]
                    else:
                        actions = np.stack([ag.sample() for ag in self.agents])

                    actions_prob = []
                    for idx, ag in enumerate(self.agents):
                        actions_prob.append(ag.get_action_prob(last_obs[idx], last_phase[idx]))

                    rewards_list = []
                    for _ in range(self.action_interval):
                        obs, rewards, dones, _ = self.env.step(actions.flatten())
                        i += 1
                        rewards_list.append(np.stack(rewards))
                    rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                    self.metric.update(rewards)

                    cur_phase = np.stack([ag.get_phase() for ag in self.agents])
                    for idx, ag in enumerate(self.agents):
                        ag.remember(last_obs[idx], last_phase[idx], actions[idx], actions_prob[idx], rewards[idx],
                            obs[idx], cur_phase[idx], dones[idx], f'{e}_{i//self.action_interval}_{ag.id}')
                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0
                        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
                    total_decision_num += 1
                    last_obs = obs
                if total_decision_num > self.learning_start and\
                        total_decision_num % self.update_model_rate == self.update_model_rate - 1:

                    cur_loss_q = np.stack([ag.train() for ag in self.agents])  # TODO: training

                    episode_loss.append(cur_loss_q)
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                    [ag.update_target_network() for ag in self.agents]

                if all(dones):
                    break
            if len(episode_loss) > 0:
                mean_loss = np.mean(np.array(episode_loss))
            else:
                mean_loss = 0
            
            self.writeLog("TRAIN", e, self.metric.real_average_travel_time(),\
                mean_loss, self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput())
            self.logger.info("step:{}/{}, q_loss:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(i, self.steps,\
                mean_loss, self.metric.rewards(), self.metric.queue(), self.metric.delay(), int(self.metric.throughput())))
            if e % self.save_rate == 0:
                [ag.save_model(e=e) for ag in self.agents]
            self.logger.info("episode:{}/{}, real avg travel time:{}".format(e, self.episodes, self.metric.real_average_travel_time()))
            for j in range(len(self.world.intersections)):
                self.logger.debug("intersection:{}, mean_episode_reward:{}, mean_queue:{}".format(j, self.metric.lane_rewards()[j],\
                     self.metric.lane_queue()[j]))
            if self.test_when_train:
                self.train_test(e)
        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
        [ag.save_model(e=self.episodes) for ag in self.agents]

    def train_test(self, e):
        '''
        train_test
        Evaluate model performance after each episode training process.

        :param e: number of episode
        :return self.metric.real_average_travel_time: travel time of vehicles
        '''
        profile_start = time.perf_counter()
        if hasattr(self.world, "reset_runtime_profile"):
            self.world.reset_runtime_profile()
        reset_start = time.perf_counter()
        obs = self.env.reset()
        reset_wall_s = time.perf_counter() - reset_start
        self.metric.clear()
        for a in self.agents:
            a.reset()
        wall_start = time.time()
        self._live_status("TRAIN_TEST", 0, self.test_steps, wall_start)
        loop_start = time.perf_counter()
        action_select_total_s = 0.0
        env_step_total_s = 0.0
        decisions = 0
        simulated_steps = 0
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []
                action_start = time.perf_counter()
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))
                action_select_total_s += time.perf_counter() - action_start
                decisions += 1
                actions = np.stack(actions)
                rewards_list = []
                step_start = time.perf_counter()
                for _ in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())  # make sure action is [intersection]
                    i += 1
                    simulated_steps += 1
                    rewards_list.append(np.stack(rewards))
                env_step_total_s += time.perf_counter() - step_start
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric.update(rewards)
                if i % self.progress_interval == 0 or i >= self.test_steps:
                    self._live_status("TRAIN_TEST", i, self.test_steps, wall_start)
            if all(dones):
                break
        loop_wall_s = time.perf_counter() - loop_start
        total_wall_s = time.perf_counter() - profile_start
        self._log_runtime_profile(
            "TRAIN_TEST",
            simulated_steps,
            decisions,
            reset_wall_s,
            loop_wall_s,
            total_wall_s,
            action_select_total_s,
            env_step_total_s,
        )
        self.logger.info("Test step:{}/{}, travel time :{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(\
            e, self.episodes, self.metric.real_average_travel_time(), self.metric.rewards(),\
            self.metric.queue(), self.metric.delay(), int(self.metric.throughput())))
        self.writeLog("TEST", e, self.metric.real_average_travel_time(),\
            100, self.metric.rewards(),self.metric.queue(),self.metric.delay(), self.metric.throughput())
        return self.metric.real_average_travel_time()

    def test(self, drop_load=True):
        '''
        test
        Test process. Evaluate model performance.

        :param drop_load: decide whether to load pretrained model's parameters.
                          False → load from disk; when load_prefix is set, load from that prefix path.
        :return self.metric: including queue length, throughput, delay and travel time
        '''
        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            if self.save_replay:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"final.txt"))
            else:
                self.env.eng.set_save_replay(False)
        self.metric.clear()
        if not drop_load:
            load_prefix = Registry.mapping['command_mapping']['setting'].param.get('load_prefix')
            if load_prefix:
                cmd = Registry.mapping['command_mapping']['setting'].param
                model_dir = os.path.join(
                    'data', 'output_data', cmd['task'],
                    f"{cmd['world']}_{cmd['agent']}",
                    cmd['network'], load_prefix, 'model'
                )
                for ag in self.agents:
                    model_file = os.path.join(model_dir, f'{self.episodes}_{ag.rank}.pt')
                    self.logger.info(f"Loading model from: {model_file}")
                    ag.load_model_from_file(model_file)
            else:
                [ag.load_model(self.episodes) for ag in self.agents]
        attention_mat_list = []
        profile_start = time.perf_counter()
        if hasattr(self.world, "reset_runtime_profile"):
            self.world.reset_runtime_profile()
        reset_start = time.perf_counter()
        obs = self.env.reset()
        reset_wall_s = time.perf_counter() - reset_start
        for a in self.agents:
            a.reset()
        wall_start = time.time()
        self._live_status("TEST", 0, self.test_steps, wall_start)
        loop_start = time.perf_counter()
        action_select_total_s = 0.0
        env_step_total_s = 0.0
        decisions = 0
        simulated_steps = 0
        for i in range(self.test_steps):
            if i % self.action_interval == 0:
                phases = np.stack([ag.get_phase() for ag in self.agents])
                actions = []
                action_start = time.perf_counter()
                for idx, ag in enumerate(self.agents):
                    actions.append(ag.get_action(obs[idx], phases[idx], test=True))
                action_select_total_s += time.perf_counter() - action_start
                decisions += 1
                actions = np.stack(actions)
                rewards_list = []
                step_start = time.perf_counter()
                for j in range(self.action_interval):
                    obs, rewards, dones, _ = self.env.step(actions.flatten())
                    i += 1
                    simulated_steps += 1
                    rewards_list.append(np.stack(rewards))
                env_step_total_s += time.perf_counter() - step_start
                rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                self.metric.update(rewards)
                if i % self.progress_interval == 0 or i >= self.test_steps:
                    self._live_status("TEST", i, self.test_steps, wall_start)
            if all(dones):
                break
        loop_wall_s = time.perf_counter() - loop_start
        total_wall_s = time.perf_counter() - profile_start
        self._log_runtime_profile(
            "TEST",
            simulated_steps,
            decisions,
            reset_wall_s,
            loop_wall_s,
            total_wall_s,
            action_select_total_s,
            env_step_total_s,
        )
        self.logger.info("Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (self.metric.real_average_travel_time(), \
            self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput()))
        self.writeLog("TEST", self.episodes,
                      self.metric.real_average_travel_time(),
                      0,
                      self.metric.rewards(),
                      self.metric.queue(),
                      self.metric.delay(),
                      self.metric.throughput())
        return self.metric

    def writeLog(self, mode, step, travel_time, loss, cur_rwd, cur_queue, cur_delay, cur_throughput):
        '''
        writeLog
        Write log for record and debug.

        :param mode: "TRAIN" or "TEST"
        :param step: current step in simulation
        :param travel_time: current travel time
        :param loss: current loss
        :param cur_rwd: current reward
        :param cur_queue: current queue length
        :param cur_delay: current delay
        :param cur_throughput: current throughput
        :return: None
        '''
        res = Registry.mapping['model_mapping']['setting'].param['name'] + '\t' + mode + '\t' + str(
            step) + '\t' + "%.1f" % travel_time + '\t' + "%.1f" % loss + "\t" +\
            "%.2f" % cur_rwd + "\t" + "%.2f" % cur_queue + "\t" + "%.2f" % cur_delay + "\t" + "%d" % cur_throughput
        log_handle = open(self.log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()

