import orhelper
from orhelper import OpenRocketInstance, Helper, FlightDataType
import numpy as np
import pandas as pd
from pathlib import Path
import itertools
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RocketDataGenerator:
    def __init__(self, jar_path: str = None):
        self.jar_path = jar_path
        
    def generate_launch_configs(self, 
                                 launch_sites: List[Tuple[float, float]],
                                 wind_speeds: List[float],
                                 wind_directions: List[float],
                                 temperatures: List[float],
                                 pressures: List[float],
                                 launch_angles: List[float],
                                 launch_directions: List[float]) -> List[Dict]:
        configs = []
        for site, wind_speed, wind_dir, temp, pressure, angle, direction in \
                itertools.product(launch_sites, wind_speeds, wind_directions, 
                                  temperatures, pressures, launch_angles, launch_directions):
            configs.append({
                'latitude': site[0],
                'longitude': site[1],
                'wind_speed': wind_speed,
                'wind_direction': wind_dir,
                'temperature': temp,
                'pressure': pressure,
                'launch_angle': angle,
                'launch_direction': direction
            })
        return configs
    
    def inject_sensor_noise(self, data: pd.DataFrame, noise_config: Dict[str, Dict]) -> pd.DataFrame:
        noisy_data = data.copy()
        
        for column, params in noise_config.items():
            if column in noisy_data.columns:
                std = params.get('std', 0.0)
                bias = params.get('bias', 0.0)
                
                # Generate noise
                noise = np.random.normal(bias, std, len(noisy_data))
                
                # Add noise to data
                noisy_data[column] = noisy_data[column] + noise
                
                logging.debug(f"Added noise to {column}: std={std}, bias={bias}")
        
        return noisy_data
    
    def run_simulation(self, ork_file: str, config: Dict, orh: Helper) -> Tuple[pd.DataFrame, Dict]:
        # Load ork
        doc = orh.load_doc(ork_file)
        sim = doc.getSimulation(0)
        opts = sim.getOptions()
        
        # Set launch site
        opts.setLaunchLatitude(config['latitude'])
        opts.setLaunchLongitude(config['longitude'])
        
        # Set atmospheric conditions
        opts.setLaunchTemperature(config['temperature'] + 273.15)  # Convert to Kelvin
        opts.setLaunchPressure(config['pressure'])
        
        # Set wind
        opts.setWindSpeedAverage(config['wind_speed'])
        opts.setWindDirection(config['wind_direction'])
        
        # Set launch rod
        opts.setLaunchRodAngle(np.radians(config['launch_angle']))
        opts.setLaunchRodDirection(np.radians(config['launch_direction']))
        
        # Run sim
        orh.run_simulation(sim)
        
        # Yoink useful vars
        variables = [
            FlightDataType.TYPE_TIME,
            FlightDataType.TYPE_AIR_TEMPERATURE,
            FlightDataType.TYPE_AIR_PRESSURE,
            FlightDataType.TYPE_ALTITUDE,
            FlightDataType.TYPE_VELOCITY_XY,
            FlightDataType.TYPE_VELOCITY_Z,
            FlightDataType.TYPE_ACCELERATION_XY,
            FlightDataType.TYPE_ACCELERATION_Z,
            FlightDataType.TYPE_ROLL_RATE,
            FlightDataType.TYPE_PITCH_RATE,
            FlightDataType.TYPE_YAW_RATE,
        ]
        
        data_dict = orh.get_timeseries(sim, variables)
        events = orh.get_events(sim)
        df = pd.DataFrame(data_dict)
        
        df.columns = [col.replace('FlightDataType.TYPE_', '').replace('TYPE_', '') for col in df.columns]

        return df, events
    
    def generate_dataset(self,
                        ork_files: List[str],
                        configs: List[Dict],
                        output_dir: str,
                        noise_config: Dict = None,
                        add_config_to_csv: bool = True):
        output_path = Path(output_dir)
        ground_truth_path = output_path / "ground_truth"
        generated_path = output_path / "generated"

        # Create both directories
        ground_truth_path.mkdir(parents=True, exist_ok=True)
        generated_path.mkdir(parents=True, exist_ok=True)

        # Noise config
        if noise_config is None:
            noise_config = {
                'ALTITUDE': {'std': 0.5, 'bias': 0.0},
                'VELOCITY_Z': {'std': 0.1, 'bias': 0.0},
                'ACCELERATION_Z': {'std': 0.5, 'bias': 0.0},
                'VELOCITY_XY': {'std': 0.1, 'bias': 0.0},
                'ACCELERATION_XY': {'std': 0.5, 'bias': 0.0},
                'ROLL_RATE': {'std': 0.05, 'bias': 0.0},
                'PITCH_RATE': {'std': 0.05, 'bias': 0.0},
                'YAW_RATE': {'std': 0.05, 'bias': 0.0},
                'AIR_TEMPERATURE': {'std': 0.1, 'bias': 0.0},
                'AIR_PRESSURE': {'std': 50.0, 'bias': 0.0},
            }
        
        total_sims = len(ork_files) * len(configs)
        logging.info(f"Starting generation of {total_sims} simulations")
        
        with OpenRocketInstance(self.jar_path) as instance:
            orh = Helper(instance)
            
            sim_count = 0
            for ork_file in ork_files:
                rocket_name = Path(ork_file).stem
                logging.info(f"Processing rocket: {rocket_name}")
                
                for idx, config in enumerate(configs):
                    try:
                        # Run and grab data
                        df, events = self.run_simulation(ork_file, config, orh)

                        # Create ground truth data
                        df_ground_truth = df.copy()
                        if add_config_to_csv:
                            for key, value in config.items():
                                df_ground_truth[f'config_{key}'] = value

                        # Create generated data
                        df_generated = self.inject_sensor_noise(df, noise_config)
                        if add_config_to_csv:
                            for key, value in config.items():
                                df_generated[f'config_{key}'] = value

                        # Save both versions
                        filename = f"{rocket_name}_sim_{idx:04d}.csv"
                        ground_truth_file = ground_truth_path / filename
                        generated_file = generated_path / filename

                        df_ground_truth.to_csv(ground_truth_file, index=False)
                        df_generated.to_csv(generated_file, index=False)

                        sim_count += 1
                        if sim_count % 10 == 0:
                            logging.info(f"Completed {sim_count}/{total_sims} simulations")
                    
                    except Exception as e:
                        logging.error(f"Failed simulation for {rocket_name} config {idx}: {e}")
                        continue
        
        logging.info(f"Dataset generation complete. Generated {sim_count} CSV files each in ground_truth and generated folders under {output_dir}")


def test():
    launch_sites = [
        (42.700473, -77.194522), # URRG
        (31.049806, -103.547306), # Midland Spaceport
        (32.990278, -106.969722), # Spaceport America
        (35.349, -117.808), # FAR Mojave Desert
        (40.883, -119.035), # Black Rock Desert
        (34.495, -116.960), # Lucerne Dry Lake
        (37.170, -97.737), # Argonia Rocket Pasture
        (35.776, -115.228), # Jean Dry Lake
    ]
    
    wind_speeds = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]  # m/s
    wind_directions = [0.0, 90.0, 180.0, 270.0]  # degrees
    temperatures = [15.0, 20.0, 25.0, 30.0, 35.0, 40.0]  # Celsius
    pressures = [101325.0]  # Standard pressure in Pa
    launch_angles = [0.0, 2.5, 5.0]  # degrees from vertical
    launch_directions = [0.0]  # degrees from North
    
    # Init
    generator = RocketDataGenerator("/home/aaron/OpenRocket/OpenRocket.jar")
    
    # Generate all configuration combinations
    configs = generator.generate_launch_configs(
        launch_sites=launch_sites,
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        temperatures=temperatures,
        pressures=pressures,
        launch_angles=launch_angles,
        launch_directions=launch_directions
    )
    
    logging.info(f"Generated {len(configs)} unique configurations")
    
    custom_noise = {
        'ALTITUDE': {'std': 1.0, 'bias': 0.2},
        'VELOCITY_Z': {'std': 0.2, 'bias': 0.0},
        'ACCELERATION_Z': {'std': 1.0, 'bias': 0.0},
    }
    
    # .ork files to process
    ork_files = [
        '/home/aaron/Development/RIT/CSCI630/finalproject/data/rockets/openrockets/L1 Kit 2025.ork',
    ]
    
    # Generate dataset
    generator.generate_dataset(
        ork_files=ork_files,
        configs=configs,
        output_dir='./simulation_data',
        noise_config=custom_noise,
        add_config_to_csv=True
    )


if __name__ == '__main__':
    test()
