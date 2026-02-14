from orhelper import OpenRocketInstance, Helper, FlightDataType
import numpy as np
import pandas as pd
from pathlib import Path
import itertools
from typing import List, Dict, Tuple
import logging
import os
import argparse
import re

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

        logging.debug(f"Events type: {type(events)}")
        logging.debug(f"Events content: {events}")
        if hasattr(events, 'keys'):
            logging.debug(f"Event keys: {list(events.keys())}")

        df.columns = [str(col).replace('FlightDataType.TYPE_', '').replace('TYPE_', '') for col in df.columns]

        return df, events
    
    def generate_dataset(self,
                        ork_files: List[str],
                        configs: List[Dict],
                        output_dir: str,
                        noise_config: Dict = None,
                        include_event_labels: bool = True,
                        include_features: bool = False):
        output_path = Path(output_dir)

        event_ground_truth_path = output_path / "event_labeled_ground_truth"
        event_ground_truth_path.mkdir(parents=True, exist_ok=True)

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
                        # Run simulation and get clean data
                        df, events = self.run_simulation(ork_file, config, orh)

                        # Create filename with configuration encoded as numbers only
                        config_parts = []
                        config_parts.append(f"{config['latitude']:.6f}")
                        config_parts.append(f"{config['longitude']:.6f}")
                        config_parts.append(f"{config['wind_speed']:.1f}")
                        config_parts.append(f"{config['wind_direction']:.0f}")
                        config_parts.append(f"{config['temperature']:.1f}")
                        config_parts.append(f"{config['launch_angle']:.1f}")
                        config_parts.append(f"{config['launch_direction']:.0f}")

                        config_string = "_".join(config_parts)
                        filename = f"{rocket_name}_{config_string}_sim_{idx:04d}.csv"

                        df_events_gt = self.process_events_for_classification(df, events)
                        event_gt_file = event_ground_truth_path / filename
                        df_events_gt.to_csv(event_gt_file, index=False)

                        sim_count += 1
                        if sim_count % 10 == 0:
                            logging.info(f"Completed {sim_count}/{total_sims} simulations")
                    
                    except Exception as e:
                        logging.error(f"Failed simulation for {rocket_name} config {idx}: {e}")
                        continue
        
        logging.info(f"Dataset generation complete. Generated {sim_count} clean CSV files in event_labeled_ground_truth folder under {output_dir}")

    def process_events_for_classification(self, df: pd.DataFrame, events: Dict, time_window: float = 1.0) -> pd.DataFrame:
        df_labeled = df.copy()

        # Initialize event labels (0 = no event, 1 = event occurring)
        event_types = ['liftoff', 'burnout', 'apogee', 'recovery_deployment', 'landing', 'stage_separation']
        for event_type in event_types:
            df_labeled[f'event_{event_type}'] = 0

        # Process events if available
        if events and isinstance(events, dict):
            logging.debug(f"Processing {len(events)} events for classification")
            for event_name, event_time in events.items():
                event_name_clean = str(event_name).lower().replace(' ', '_')
                logging.debug(f"Event: {event_name_clean}, Time: {event_time}, Type: {type(event_time)}")

                # Map OpenRocket event names to our standardiczed names
                event_mapping = {
                    'ignition': 'liftoff',
                    'launch': 'liftoff',
                    'motor_burnout': 'burnout',
                    'burnout': 'burnout',
                    'apogee': 'apogee',
                    'recovery_device_deployment': 'recovery_deployment',
                    'ejection_charge': 'recovery_deployment',
                    'landing': 'landing',
                    'ground_hit': 'landing',
                    'stage_separation': 'stage_separation'
                }

                # Find matching event type
                mapped_event = None
                for key, value in event_mapping.items():
                    if key in event_name_clean:
                        mapped_event = value
                        break

                if mapped_event and f'event_{mapped_event}' in df_labeled.columns:
                    # Handle case where event_time might be a list or single value
                    if isinstance(event_time, list):
                        event_times = event_time
                    else:
                        event_times = [event_time]

                    # Process each event time
                    for single_event_time in event_times:
                        try:
                            # Ensure single_event_time is a number
                            if not isinstance(single_event_time, (int, float)):
                                logging.warning(f"Skipping non-numeric event time: {single_event_time} (type: {type(single_event_time)})")
                                continue

                            # Mark time points within window of event
                            time_mask = (df_labeled['TIME'] >= single_event_time - time_window/2) & \
                                       (df_labeled['TIME'] <= single_event_time + time_window/2)
                            df_labeled.loc[time_mask, f'event_{mapped_event}'] = 1


                            logging.debug(f"Processed event {mapped_event} at time {single_event_time}")
                        except Exception as e:
                            logging.warning(f"Error processing event {mapped_event} at time {single_event_time}: {e}")
                            continue

        return df_labeled

    @staticmethod
    def parse_config_from_filename(filename: str) -> Dict:

        config = {}

        # Extract rocket name and parameters using regex
        # Format: RocketName_lat_lon_ws_wd_temp_la_ld_sim_XXXX.csv
        pattern = r'(.+)_(-?\d+\.\d+)_(-?\d+\.\d+)_(\d+\.\d+)_(\d+)_(\d+\.\d+)_(\d+\.\d+)_(\d+)_sim_\d+\.csv'
        match = re.match(pattern, filename)

        if match:
            config = {
                'rocket_name': match.group(1),
                'latitude': float(match.group(2)),
                'longitude': float(match.group(3)),
                'wind_speed': float(match.group(4)),
                'wind_direction': float(match.group(5)),
                'temperature': float(match.group(6)),
                'launch_angle': float(match.group(7)),
                'launch_direction': float(match.group(8))
            }

        return config

def generate(openrocket_jar_path: str, input_dir: str, output_dir: str):
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
    generator = RocketDataGenerator(jar_path=openrocket_jar_path)
    
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
    
    # .ork files to process
    ork_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.ork')]

    # Generate dataset with only clean event classification data
    generator.generate_dataset(
        ork_files=ork_files,
        configs=configs,
        output_dir=output_dir,
        include_event_labels=True,
        include_features=False
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate rocket flight datasets from OpenRocket simulations.')
    parser.add_argument('--jar_path', type=str, default=None, help='Path to OpenRocket JAR file (optional if in classpath)')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing .ork files')
    parser.add_argument('--output_dir', type=str, default='./simulation_data', help='Directory to save generated datasets')

    args = parser.parse_args()

    generate(
        openrocket_jar_path=args.jar_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
