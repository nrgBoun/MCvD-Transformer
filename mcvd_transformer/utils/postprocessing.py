import os
import numpy as np
import cv2


class PostProcessing:

    @staticmethod
    def moving_average(signal, n):
        result = np.zeros(signal.shape)
        temp_signal = np.insert(signal, 0, [0 for i in range(n - 1)])
        for i in range(signal.shape[0]):
            temp_sum = sum([temp_signal[i + j] for j in range(n)])
            result[i] = temp_sum / n
        return result

    @staticmethod
    def post_processing(signal):
        noisy_pdf = np.array(
            [signal[i + 1] - signal[i] for i in range(signal.shape[0] - 1)]
        )
        noisy_pdf[noisy_pdf < 0] = 0
        clean_pdf = PostProcessing.moving_average(noisy_pdf, 3)
        result_array = np.zeros(signal.shape)
        for i in range(clean_pdf.shape[0]):
            result_array[i + 1] = result_array[i] + clean_pdf[i]
        return np.add(result_array, signal[0])

    @staticmethod
    def postprocessing_separate(prediction_shape, prediction_max):
        postprocessed_prediction = PostProcessing.post_processing(prediction_shape)[10:]
        postprocessed_prediction_ratio = (
            postprocessed_prediction / postprocessed_prediction[-1]
        )
        postprocessed_prediction = postprocessed_prediction_ratio * prediction_max
        return postprocessed_prediction

    @staticmethod
    def average_merging(array1, array2):
        # Crop First part of array 1
        array1 = array1[10:]
        # Convert arrays to pdf
        pdf_array1 = np.diff(array1)
        pdf_array2 = np.diff(array2)
        # pdf_array3 = np.diff(array3)
        # Average Merge Areas
        between1 = (pdf_array1[-9:] + pdf_array2[:9]) / 2
        # between2 = (pdf_array2[-9:] + pdf_array3[:9])/2
        merged_pdf = np.concatenate((pdf_array1[:-9], between1, pdf_array2[9:]))
        return np.hstack((0, merged_pdf)).cumsum()

    @staticmethod
    def remove_noise(signal, n):
        noisy_pdf = np.array(
            [signal[i + 1] - signal[i] for i in range(signal.shape[0] - 1)]
        )
        clean_pdf = PostProcessing.moving_average(noisy_pdf, n)
        result_array = np.zeros(signal.shape)
        for i in range(clean_pdf.shape[0]):
            result_array[i + 1] = result_array[i] + clean_pdf[i]
        return result_array

    @staticmethod
    def predict(model, test_data, base_directory, scale_index):

        angle = sorted(list(test_data["scale_dict"].keys()))[scale_index]
        image_path = (
            base_directory
            + os.path.sep
            + test_data["image_path"]
            + os.path.sep
            + f"{angle}.jpg"
        )
        image_input = cv2.imread(image_path) / 255.0
        image_input = np.expand_dims(image_input, axis=0)

        number_input = np.array(
            [
                test_data["scale_dict"][angle],
                test_data["time_step"],
                test_data["diff_coef"],
            ]
        )
        number_input = np.expand_dims(number_input, axis=0)
        result = model.predict([number_input, image_input])

        return result

    @staticmethod
    def predict_shape_max_combined(model, test_data, base_directory, scale_index):

        prediction = PostProcessing.predict(
            model, test_data, base_directory, scale_index
        )
        return prediction[0][0], prediction[1][0]
