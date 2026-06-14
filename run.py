import task
import trainer
import agent
import dataset
from common.registry import Registry
from common import interface
from common.utils import *
from utils.logger import *
import time
from datetime import datetime
import argparse


# parseargs
parser = argparse.ArgumentParser(description='Run Experiment')
parser.add_argument('--thread_num', type=int, default=4, help='number of threads')  # used in cityflow
parser.add_argument('--ngpu', type=str, default="-1", help='gpu to be used')  # choose gpu card
parser.add_argument('--prefix', type=str, default='test', help="the number of prefix in this running process")
parser.add_argument('--seed', type=int, default=None, help="seed for pytorch backend")
parser.add_argument('--debug', type=bool, default=True)
parser.add_argument('--interface', type=str, default="libsumo", choices=['libsumo','traci'], help="interface type") # libsumo(fast) or traci(slow)
parser.add_argument('--delay_type', type=str, default="apx", choices=['apx','real'], help="method of calculating delay") # apx(approximate) or real

parser.add_argument('-t', '--task', type=str, default="tsc", help="task type to run")
parser.add_argument('-a', '--agent', type=str, default="dqn", help="agent type of agents in RL environment")
parser.add_argument('-w', '--world', type=str, default="cityflow", choices=['cityflow','sumo'], help="simulator type")
parser.add_argument('-n', '--network', type=str, default="cityflow1x1", help="network name")
parser.add_argument('-d', '--dataset', type=str, default='onfly', help='type of dataset in training process')

# Partial observability and OMNeT co-simulation arguments
parser.add_argument('--episodes', type=int, default=None,
                    help="override the number of training episodes (uses config default if omitted)")
parser.add_argument('--train_model', type=lambda x: x.lower() not in ('false', '0', 'no'),
                    default=None, metavar='BOOL',
                    help="override train_model from config YAML (true/false)")
parser.add_argument('--test_model', type=lambda x: x.lower() not in ('false', '0', 'no'),
                    default=None, metavar='BOOL',
                    help="override test_model from config YAML (true/false)")
parser.add_argument('--load_prefix', type=str, default=None,
                    help="prefix of a previously trained run whose model weights to load for inference "
                         "(e.g. 'omnet_off__pr_1.00'); model is loaded from that run's model/ folder")
parser.add_argument('--penetration_rate', type=float, default=1.0,
                    help="vehicle penetration rate for partial observability [0.0-1.0]; "
                         "1.0 = full observability, 0.5 = 50%% of vehicles visible")
parser.add_argument('--use_omnet', action='store_true', default=False,
                    help="use OMNeT++ CSV output to filter observable vehicles "
                         "(overrides penetration_rate-based sampling with CSV vehicle list)")
parser.add_argument('--omnet_csv_path', type=str,
                    default='/home/exx/Desktop/vtc2026/omnet_files/gwu-workspace-pedestrians/'
                            'simu5G/simulations/NR/cars/SUMO_output_CV2X.csv',
                    help="path to the OMNeT SUMO_output CSV file containing visible vehicle IDs")
parser.add_argument('--traci_port', type=int, default=9999,
                    help="TraCI port for SUMO when --interface traci (must match OMNeT veinsManager.port)")
parser.add_argument('--traci_connect_retries', type=int, default=120,
                    help="When using traci, retry connecting this many times while waiting for OMNeT")
parser.add_argument('--traci_connect_delay', type=float, default=1.0,
                    help="Seconds between TraCI connection retries")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.ngpu

logging_level = logging.INFO
if args.debug:
    logging_level = logging.DEBUG


class Runner:
    def __init__(self, pArgs):
        """
        instantiate runner object with processed config and register config into Registry class
        """
        self.config, self.duplicate_config = build_config(pArgs)
        self.config_registry()

    def config_registry(self):
        """
        Register config into Registry class
        """

        # Allow CLI --episodes to override the YAML trainer setting
        if self.config['command'].get('episodes') is not None:
            self.config['trainer']['episodes'] = self.config['command']['episodes']

        # Allow CLI --train_model / --test_model to override YAML model settings
        if self.config['command'].get('train_model') is not None:
            self.config['model']['train_model'] = self.config['command']['train_model']
        if self.config['command'].get('test_model') is not None:
            self.config['model']['test_model'] = self.config['command']['test_model']

        interface.Command_Setting_Interface(self.config)
        interface.Logger_param_Interface(self.config)  # register logger path
        interface.World_param_Interface(self.config)
        if self.config['model'].get('graphic', False):
            param = Registry.mapping['world_mapping']['setting'].param
            if self.config['command']['world'] in ['cityflow', 'sumo']:
                roadnet_path = param['dir'] + param['roadnetFile']
            else:
                roadnet_path = param['road_file_addr']
            interface.Graph_World_Interface(roadnet_path)  # register graphic parameters in Registry class
        interface.Logger_path_Interface(self.config)
        # make output dir if not exist
        if not os.path.exists(Registry.mapping['logger_mapping']['path'].path):
            os.makedirs(Registry.mapping['logger_mapping']['path'].path)        
        interface.Trainer_param_Interface(self.config)
        interface.ModelAgent_param_Interface(self.config)

    def run(self):
        logger = setup_logging(logging_level)
        self.trainer = Registry.mapping['trainer_mapping']\
            [Registry.mapping['command_mapping']['setting'].param['task']](logger)
        self.task = Registry.mapping['task_mapping']\
            [Registry.mapping['command_mapping']['setting'].param['task']](self.trainer)
        start_time = time.time()
        self.task.run()
        logger.info(f"Total time taken: {time.time() - start_time}")


if __name__ == '__main__':
    test = Runner(args)
    test.run()

