import pandas as pd

class Dataloader:
    SENSOR_COLS = ["AIR_PRESSURE", "AIR_TEMPERATURE", "ACCELERATION_XY", "ACCELERATION_Z"]
    EVENT_COLS = ["event_liftoff"] # TODO: Check the data

    def __init__(self, data_path, sequence_length=100, stride=50, test_size=0.2):
        self.data_path = data_path
        self.data = None
        self.sequence_length = sequence_length
        self.stride = stride
        self.test_size = test_size

    def load_chunk(self, chunk_size=100):
        csv_files = list(self.data_path.glob("*.csv"))

        for i in range(0, len(csv_files), chunk_size):
            chunk_files = csv_files[i:i+chunk_size]
            chunk_X = []
            chunk_y = []

            for file in chunk_files:
                df = pd.read_csv(file)
                X = df[self.SENSOR_COLS].values
                y = df[self.EVENT_COLS].values
                chunk_X.append(X)
                chunk_y.append(y)

            if chunk_X:
                yield chunk_X, chunk_y
