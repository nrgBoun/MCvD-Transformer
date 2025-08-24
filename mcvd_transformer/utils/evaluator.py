import os
from tqdm import tqdm
import json
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image, display
from scipy import stats
from .postprocessing import PostProcessing


class PerformanceEvaluator:

    @staticmethod
    def rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    @staticmethod
    def find_peak(signal, n):
        return int(np.mean(np.diff(signal).argsort()[-n:]))

    @staticmethod
    def calculate_peak_list(val_set, outputs):

        peak_list = []
        for x in val_set:
            peak_list.append(
                PerformanceEvaluator.find_peak(
                    outputs[x["image_path"]]["time_output"], 3
                )
            )

        return sorted([(peak_list[i], i) for i in range(len(peak_list))], reverse=False)

    @staticmethod
    def calculate_std_list(val_set, outputs):

        std_list = []
        for x in val_set:
            std_list.append(
                np.mean(
                    np.std(
                        PerformanceEvaluator.rolling_window(
                            outputs[x["image_path"]]["time_output"], 5
                        ),
                        axis=1,
                    )
                )
            )

        return sorted([(std_list[i], i) for i in range(len(std_list))], reverse=False)

    @staticmethod
    def findIndex(
        number_of_absorbers, number_of_reflectors, data_set, error_values=None
    ):
        temp_list = []
        for i, x in enumerate(data_set):
            if (
                len(x.absorbers) == number_of_absorbers - 1
                and len(x.reflectors) == number_of_reflectors
            ):
                row = [
                    i,
                    f"{number_of_absorbers} Absorbers - {number_of_reflectors} Reflectors",
                ]
                if error_values is not None:
                    row += [error_values[-2], error_values[-1]]
                temp_list.append(row)

        return temp_list

    @staticmethod
    def angle_graph(angle_output, image_loc, path=None):
        fig = plt.figure(figsize=(12, 6))
        N = 180
        theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        radii = []
        for i in range(N):
            sum = 0
            for j in range(int(360 / N)):
                sum += angle_output[int(360 / N) * i + j]
            radii.append(sum)
        radii = np.array(radii)
        width = (2 * np.pi / N) * np.ones(N)
        colors = plt.cm.viridis(radii / 10.0)

        ax1 = plt.subplot(1, 2, 1, projection="polar")
        ax1.set_yticklabels([])
        ax1.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.5)

        image = mpimg.imread(image_loc)

        ax2 = plt.subplot(1, 2, 2)
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])

        if path is not None:
            plt.savefig(path + os.path.sep + "angle_plot.png")

    @staticmethod
    def show_image(image_path, width=200, height=200):
        pil_img = Image(filename=image_path, width=width, height=height)
        display(pil_img)

    @staticmethod
    def ks_test(expected_signal, predicted_signal):
        ks_result = stats.ks_2samp(expected_signal, predicted_signal)
        print("KS Statistic = {}, P-Value = {}".format(ks_result[0], ks_result[1]))
        return ks_result[0], ks_result[1]

    @staticmethod
    def time_graph(
        time_output_actual,
        time_output_predicted_list,
        legend=["Actual", "Predicted"],
        image_loc=None,
        time_res=1,
        expension_ratio=1,
        path=None,
    ):

        time_output_actual = [
            np.mean(time_output_actual[i * time_res : (i + 1) * time_res])
            for i in range(int(len(time_output_actual) / time_res))
        ]
        time_output_predicted_list = [
            [
                np.mean(time_output_predicted.T[i * time_res : (i + 1) * time_res])
                for i in range(int(time_output_predicted.shape[0] / time_res))
            ]
            for time_output_predicted in time_output_predicted_list
        ]

        plt.figure(figsize=(24, 8))

        ax1 = plt.subplot(1, 3, 1)
        plt.plot(time_output_actual)
        for time_output_predicted in time_output_predicted_list:
            plt.plot(time_output_predicted, linestyle="dashed")

        if expension_ratio != 1:
            ax1.set_ylim((0, time_output_actual[-1] * expension_ratio))

        plt.ylim(0, 0.3)
        plt.xlabel("Time Step")
        plt.ylabel("Ratio of Absorbed Molecules (Cumulative)")
        plt.legend(legend)

        ax2 = plt.subplot(1, 3, 2)
        plt.plot(
            [
                time_output_actual[i + 1] - time_output_actual[i]
                for i in range(len(time_output_actual) - 1)
            ]
        )
        for time_output_predicted in time_output_predicted_list:
            plt.plot(
                [
                    time_output_predicted[i + 1] - time_output_predicted[i]
                    for i in range(len(time_output_predicted) - 1)
                ],
                linestyle="dashed",
            )

        plt.xlabel("Time Step")
        plt.ylabel("Ratio of Absorbed Molecules")
        plt.legend(legend)

        if path is not None:
            plt.savefig(path)

    @staticmethod
    def case_name(image_path):
        # Find the name of the case
        dir_list = image_path.split("/")
        case_splitted = dir_list[1].split("_")
        case_splitted.pop(0)
        case_splitted.pop(0)
        if "v" in case_splitted[-1]:
            case_splitted.pop(-1)
        return "_".join(case_splitted)

    @staticmethod
    def create_evaluation_results(
        model, data_set, coordinate_system, max_number_of_spherical, order, path
    ):

        np.random.seed(10)

        error_values = []

        for index in tqdm(range(len(data_set))):

            topology = data_set[index].rotate(
                np.random.randint(0, 360), np.random.randint(-90, 90)
            )

            input_top, input_num = topology.convert_numpy(
                coordinate_system, max_number_of_spherical, order, False
            )
            shape, max_value, _ = model.predict(
                [np.expand_dims(input_top, axis=0), np.expand_dims(input_num, axis=0)]
            )
            prediction = PostProcessing.postprocessing_separate(shape[0], max_value[0])

            postprocessed_shape = prediction / prediction[-1]

            # Shape Errors
            shape_avg = np.mean(
                np.abs(
                    postprocessed_shape
                    - data_set[index].time_output / data_set[index].time_output[-1]
                )
            )
            shape_max = np.max(
                np.abs(
                    postprocessed_shape
                    - data_set[index].time_output / data_set[index].time_output[-1]
                )
            )
            # Max Number Errors
            max_number_percent = (
                np.abs(max_value - data_set[index].time_output[-1])
                / data_set[index].time_output[-1]
            )
            # Total Signal Error
            total_signal_avg = np.mean(
                np.abs(
                    prediction / data_set[index].time_output[-1]
                    - data_set[index].time_output / data_set[index].time_output[-1]
                )
            )
            total_signal_max = np.max(
                np.abs(
                    prediction / data_set[index].time_output[-1]
                    - data_set[index].time_output / data_set[index].time_output[-1]
                )
            )
            # Store all Error Values
            error_values.append(
                (
                    shape_avg,
                    shape_max,
                    float(max_number_percent),
                    total_signal_avg,
                    total_signal_max,
                )
            )

        with open(path, "w") as fp:
            json.dump(error_values, fp)

        return error_values
