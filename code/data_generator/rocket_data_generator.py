import orhelper
from orhelper import OpenRocketInstance, Helper
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
            orh.FlightDataType.TYPE_TIME,
            orh.FlightDataType.TYPE_ALTITUDE,
            orh.FlightDataType.TYPE_VELOCITY_Z,
            orh.FlightDataType.TYPE_ACCELERATION_Z,
            orh.FlightDataType.TYPE_POSITION_X,
            orh.FlightDataType.TYPE_POSITION_Y,
            orh.FlightDataType.TYPE_VELOCITY_XY,
            orh.FlightDataType.TYPE_ACCELERATION_XY,
            orh.FlightDataType.TYPE_MACH_NUMBER,
            orh.FlightDataType.TYPE_ORIENTATION_THETA,
            orh.FlightDataType.TYPE_ORIENTATION_PHI,
            orh.FlightDataType.TYPE_ANGULAR_VELOCITY_ROLL,
            orh.FlightDataType.TYPE_ANGULAR_VELOCITY_PITCH,
            orh.FlightDataType.TYPE_ANGULAR_VELOCITY_YAW,
            orh.FlightDataType.TYPE_MASS,
            orh.FlightDataType.TYPE_THRUST_FORCE,
            orh.FlightDataType.TYPE_DRAG_FORCE,
        ]
        
        data_dict = orh.get_timeseries(sim, variables)
        events = orh.get_events(sim)
        df = pd.DataFrame(data_dict)
        
        return df, events
    
    def generate_dataset(self,
                        ork_files: List[str],
                        configs: List[Dict],
                        output_dir: str,
                        noise_config: Dict = None,
                        add_config_to_csv: bool = True):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Noise config
        if noise_config is None:
            noise_config = {
                'Altitude': {'std': 0.5, 'bias': 0.0},
                'Vertical velocity': {'std': 0.1, 'bias': 0.0},
                'Vertical acceleration': {'std': 0.5, 'bias': 0.0},
                'Position X': {'std': 1.0, 'bias': 0.0},
                'Position Y': {'std': 1.0, 'bias': 0.0},
                'Lateral velocity': {'std': 0.1, 'bias': 0.0},
                'Roll rate': {'std': 0.05, 'bias': 0.0},
                'Pitch rate': {'std': 0.05, 'bias': 0.0},
                'Yaw rate': {'std': 0.05, 'bias': 0.0},
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
                        df_noisy = self.inject_sensor_noise(df, noise_config)
                        if add_config_to_csv:
                            for key, value in config.items():
                                df_noisy[f'config_{key}'] = value

                        # Save
                        output_file = output_path / f"{rocket_name}_sim_{idx:04d}.csv"
                        df_noisy.to_csv(output_file, index=False)
                        
                        sim_count += 1
                        if sim_count % 10 == 0:
                            logging.info(f"Completed {sim_count}/{total_sims} simulations")
                    
                    except Exception as e:
                        logging.error(f"Failed simulation for {rocket_name} config {idx}: {e}")
                        continue
        
        logging.info(f"Dataset generation complete. Generated {sim_count} CSV files in {output_dir}")


def test():
    launch_sites = [
        (40.7128, -74.0060),
    ]
    
    wind_speeds = [0.0, 2.0, 5.0, 10.0]  # m/s
    wind_directions = [0.0, 90.0, 180.0, 270.0]  # degrees
    temperatures = [15.0, 20.0, 25.0]  # Celsius
    pressures = [101325.0]  # Standard pressure in Pa
    launch_angles = [0.0, 2.0, 5.0]  # degrees from vertical
    launch_directions = [0.0]  # degrees from North
    
    # Init
    generator = RocketDataGenerator()
    
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
        'Altitude': {'std': 1.0, 'bias': 0.2},
        'Vertical velocity': {'std': 0.2, 'bias': 0.0},
        'Vertical acceleration': {'std': 1.0, 'bias': 0.0},
    }
    
    # .ork files to process
    ork_files = [
        'TODO.ork',
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
