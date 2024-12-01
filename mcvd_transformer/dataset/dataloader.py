from tensorflow.keras.utils import Sequence  # type: ignore
import numpy as np


class BatchGenerator(Sequence):

    def __init__(
        self,
        input_data,
        batch_size,
        coordinate_system,
        random_rotate=True,
        entity_order="normal",
        zero_padding=10,
        max_shape=True,
        shuffle=False,
        max_spherical_entity=15,
        flatten=True,
        one_absorber_points=0,
        random_seed=50,
    ):

        self.input = input_data
        self.batch_size = batch_size
        self.coordinate_system = coordinate_system
        self.random_rotate = random_rotate
        self.entity_order = entity_order
        self.zero_padding = zero_padding
        self.max_shape = max_shape
        self.shuffle = shuffle
        self.max_spherical_entity = max_spherical_entity
        self.flatten = flatten
        self.one_absorber_points = one_absorber_points
        self.current_index = 0
        np.random.seed(random_seed)

    def __len__(self):
        return (np.ceil(len(self.input) / float(self.batch_size))).astype(int)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index < len(self):
            batch = self.__getitem__(self.current_index)
            self.current_index += 1
            return batch
        else:
            self.current_index = 0  # Reset index for the next iteration
            if self.shuffle:
                np.random.shuffle(self.input)
            raise StopIteration

    def __getitem__(self, idx):

        # Take Input Batch
        batch_x = self.input[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x_los = [el.line_of_sight for el in batch_x]

        # Genrate Time Output Batch
        batch_y_time = np.array([x.time_output for x in batch_x])
        # Apply Zero Padding
        batch_y_time = np.concatenate(
            (np.zeros((batch_y_time.shape[0], self.zero_padding)), batch_y_time), axis=1
        )
        if self.max_shape:
            # Calculate Shape of the Signals
            signal_shapes = batch_y_time / batch_y_time[:, -1][:, None]
            # Calculate Maximum Values
            max_values = batch_y_time[:, -1].reshape(-1, 1)
            # Batch Output
            batch_output = [
                np.asarray(signal_shapes).astype(np.float32),
                np.asarray(max_values).astype(np.float32),
                np.asarray(batch_y_time).astype(np.float32),
            ]
        else:
            # Batch Output
            batch_output = [
                np.asarray(batch_y_time).astype(np.float32),
                np.asarray(batch_y_time).astype(np.float32),
            ]

        if self.random_rotate:
            batch_x = [
                x.rotate(np.random.randint(0, 360), np.random.randint(-90, 90))
                for x in batch_x
            ]

        if self.flatten:
            # Batch Input
            batch_input = np.asarray(
                [
                    x.convert_numpy(
                        self.coordinate_system,
                        self.max_spherical_entity,
                        self.entity_order,
                        self.flatten,
                        self.one_absorber_points,
                    )
                    for x in batch_x
                ]
            ).astype(np.float32)
        else:
            topology_inputs = []
            numerical_inputs = []
            for x, los in zip(batch_x, batch_x_los):
                numpy_repr = x.convert_numpy(
                    self.coordinate_system,
                    self.max_spherical_entity,
                    self.entity_order,
                    self.flatten,
                    self.one_absorber_points,
                )
                topology_inputs.append(numpy_repr[0])
                numerical_inputs.append(
                    np.hstack(
                        (
                            numpy_repr[1],
                            np.array(
                                [
                                    los[0] / los[-1],
                                    los[1] / los[-1],
                                    los[2] / los[-1],
                                    los[3] / los[-1],
                                ]
                            ),
                        )
                    )
                )

            batch_input = [
                np.asarray(topology_inputs).astype(np.float32),
                np.asarray(numerical_inputs).astype(np.float32),
            ]

        if idx == len(self) - 1 and self.shuffle:
            np.random.shuffle(self.input)

        return tuple(batch_input), tuple(batch_output)
