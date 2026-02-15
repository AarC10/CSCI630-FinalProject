import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
from pathlib import Path


class FlightDataVisualizer:
    def __init__(self):
        self.event_colors = {
            'event_liftoff': 'red',
            'event_burnout': 'orange',
            'event_apogee': 'green',
            'event_recovery_deployment': 'blue',
            'event_landing': 'purple',
            'event_stage_separation': 'brown'
        }

    def load_csv(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        return df

    def plot_flight_data(self, df: pd.DataFrame, save_path: str = None):
        data_columns = [
            'TIME', 'AIR_TEMPERATURE', 'AIR_PRESSURE', 'ALTITUDE',
            'VELOCITY_XY', 'VELOCITY_Z', 'ACCELERATION_XY', 'ACCELERATION_Z',
            'ROLL_RATE', 'PITCH_RATE', 'YAW_RATE'
        ]

        available_columns = [col for col in data_columns if col in df.columns]
        if 'TIME' not in available_columns:
            raise ValueError("TIME column not found in CSV")

        n_plots = len(available_columns) - 1
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        time = df['TIME']

        # Plot each var
        plot_configs = [
            ('AIR_TEMPERATURE', 'Air Temperature (K)', 'tab:red'),
            ('AIR_PRESSURE', 'Air Pressure (Pa)', 'tab:orange'),
            ('ALTITUDE', 'Altitude (m)', 'tab:green'),
            ('VELOCITY_XY', 'Velocity XY (m/s)', 'tab:blue'),
            ('VELOCITY_Z', 'Velocity Z (m/s)', 'tab:cyan'),
            ('ACCELERATION_XY', 'Acceleration XY (m/s²)', 'tab:purple'),
            ('ACCELERATION_Z', 'Acceleration Z (m/s²)', 'tab:pink'),
            ('ROLL_RATE', 'Roll Rate (rad/s)', 'tab:brown'),
            ('PITCH_RATE', 'Pitch Rate (rad/s)', 'tab:gray'),
            ('YAW_RATE', 'Yaw Rate (rad/s)', 'tab:olive')
        ]

        plot_idx = 0
        for col, ylabel, color in plot_configs:
            if col in available_columns:
                axes[plot_idx].plot(time, df[col], color=color, linewidth=1.5)
                axes[plot_idx].set_ylabel(ylabel)
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1

        # Add event vert marker lines
        event_columns = [col for col in df.columns if col.startswith('event_')]

        for event_col in event_columns:
            if event_col in self.event_colors:
                # Find times where event occurs (value = 1)
                event_times = df[df[event_col] == 1]['TIME'].values

                if len(event_times) > 0:
                    # Get the middle time of the event window
                    event_time = np.median(event_times)

                    for ax in axes:
                        ax.axvline(x=event_time,
                                   color=self.event_colors[event_col],
                                   linestyle='--',
                                   alpha=0.7,
                                   linewidth=2,
                                   label=event_col.replace('event_', '').replace('_', ' ').title())

        # Set common labels and title
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle('Rocket Flight Data with Event Markers', fontsize=16, fontweight='bold')

        # Add legend for events (only on the first subplot to avoid clutter)
        if event_columns:
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:  # Only add legend if there are event markers
                fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def print_event_summary(self, df: pd.DataFrame):
        event_columns = [col for col in df.columns if col.startswith('event_')]

        print("\nEvent Summary:")
        print("=" * 40)

        for event_col in event_columns:
            event_times = df[df[event_col] == 1]['TIME'].values
            if len(event_times) > 0:
                min_time = np.min(event_times)
                max_time = np.max(event_times)
                median_time = np.median(event_times)
                print(f"{event_col.replace('event_', '').replace('_', ' ').title()}: "
                      f"{median_time:.2f}s (window: {min_time:.2f}-{max_time:.2f}s)")
            else:
                print(f"{event_col.replace('event_', '').replace('_', ' ').title()}: No events detected")


def main():
    parser = argparse.ArgumentParser(description='Visualize rocket flight data CSV files with event markers')
    parser.add_argument('csv_path', type=str, help='Path to CSV file to visualize')
    parser.add_argument('--save', type=str, default=None, help='Path to save the plot (optional)')

    args = parser.parse_args()

    if not Path(args.csv_path).exists():
        print(f"Error: CSV file {args.csv_path} does not exist")
        return

    visualizer = FlightDataVisualizer()

    try:
        df = visualizer.load_csv(args.csv_path)
        visualizer.print_event_summary(df)
        visualizer.plot_flight_data(df, args.save)
    except Exception as e:
        print(f"Error processing CSV file: {e}")


if __name__ == '__main__':
    main()
