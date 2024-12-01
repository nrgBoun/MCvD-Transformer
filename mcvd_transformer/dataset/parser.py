import os
from tqdm import tqdm
import numpy as np
import json
import pickle

from ..utils.objects import Topology


class DataParser:

    def __init__(
        self,
        base_directory,
        unwanted_folders=["Validation Data"],
        include_prism=False,
        additional_controls=[],
    ):

        self.base_directory = base_directory
        self.unwanted_folders = unwanted_folders
        self.include_prism = include_prism
        self.additional_controls = additional_controls

    def list_simulations(self):
        simulation_list = []

        for simulation_folder in os.listdir(self.base_directory):
            if simulation_folder in self.unwanted_folders:
                continue
            for case_folder in os.listdir(
                self.base_directory + os.path.sep + simulation_folder
            ):
                if "prism" in case_folder and (not self.include_prism):
                    continue
                case_folder_path = simulation_folder + os.path.sep + case_folder
                for repetition_num in os.listdir(
                    self.base_directory + os.path.sep + case_folder_path
                ):
                    repetition_path = case_folder_path + os.path.sep + repetition_num
                    additional_control_results = [
                        control(self.base_directory + os.path.sep + repetition_path)
                        for control in self.additional_controls
                    ]
                    if False in additional_control_results:
                        continue
                    simulation_list.append(
                        case_folder_path + os.path.sep + repetition_num
                    )

        return sorted(simulation_list)

    def parse_data(
        self, enable_time_output=True, enable_angle_output=True, time_step_threshold=0
    ):

        simulation_list = self.list_simulations()

        data_set = []

        for simulation in tqdm(simulation_list):
            # Simulation Path
            simulation_path = self.base_directory + os.path.sep + simulation
            # Parse Channel Config
            channel_config = json.load(
                open(simulation_path + os.path.sep + "topology.json")
            )
            # Read Time Output
            if enable_time_output:
                time_outputs = np.load(
                    simulation_path + os.path.sep + "time_output.npy"
                )
            for i in range(len(channel_config["absorbers"])):

                # Generate Topology Instance
                topology = Topology.parse_channel_config(channel_config, i)

                # Control Time Step Threshold
                if time_step_threshold > topology.time_step:
                    continue

                # Store Time Output
                if enable_time_output:
                    topology.set_time_output(time_outputs[i])

                topology.data_path = simulation_path
                topology.intended_index = i

                # Add topology to data set
                data_set.append(topology)

        return data_set

    @staticmethod
    def save_file(data, path):
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load_file(path):
        return pickle.load(open(path, "rb"))
