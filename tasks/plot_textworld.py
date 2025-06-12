import os
import sys
from tkinter import filedialog
import logging
from datetime import datetime
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import utilities

utilities.install('matplotlib')
utilities.install('tkinter')
import matplotlib.pyplot as plt
import tkinter as tk


def open_file_dialog():
    """Opens a file dialog to select multiple files."""
    root = tk.Tk()
    root.withdraw()

    file_paths = filedialog.askopenfilenames(
        title="Select Rollout files",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )

    if file_paths:
        for file_path in file_paths:
            yield file_path


def save_file_dialog(default_path = None):
    """Opens a file dialog to save a file."""
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.asksaveasfilename(
        title="Save Plot",
        defaultextension=".pgf",
        filetypes=[("PGF Files", "*.pgf"), ("All Files", "*.*")],
        initialfile=default_path if default_path else "plot.pgf"
    )
    if file_path:
        return file_path
    else:
        logging.warning("No file selected for saving.")
        return None


def parse_rollout_file(file_generator, ma_alpha, training_trend):
    """Parses a rollout file and returns the content."""
    for file_path in file_generator:
        with open(file_path, 'r', encoding="utf-8") as file:
            content = file.read()

        # first detect session prefix "====..."
        for session_text in content.split("========================================================================"):
            session_text = session_text.strip()
            if len(session_text) < 10:
                continue
            metadata, stats = parse_session(session_text)
            if (training_trend and len(stats) > 100) or (not training_trend and len(stats) <= 100):
                yield {
                    'file_name': os.path.basename(file_path),
                    'file_path': file_path,
                    'metadata': metadata,
                    'stats': stats,
                    'moving_average': compute_moving_average(stats, alpha=ma_alpha)
                }
            else:
                logging.warning(f"Skipping session in {file_path} due to {training_trend} and {len(stats)}")


def parse_session(session):
    """
    Date: 2025-06-01 13:45:52
    Allow relegation: True
    Allow sub training: True
    Relegation probability: 0.1
    ------------------------------------------------------------------------
    Episode 10
    [Report]	episode: 10/20000; steps: 100.0; succeeded: 0.8; score: -2.0/ 1.2; cl: 100.0; max cl: 100.0
    ...
    ------------------------------------------------------------------------
    Episode 20
    ...

    """
    episodes = session.strip().split('------------------------------------------------------------------------')
    header = episodes[0].strip()
    logging.info(f"Parsing session header: {header}")
    parts = header.split('\n')
    date = utilities.get_datetime_from_string(parts[0].split(':', 1)[1].strip())
    allow_relegation = parts[1].split(':', 1)[1].strip().lower() == 'true'
    allow_sub_training = parts[2].split(':', 1)[1].strip().lower() == 'true'
    relegation_probability = float(parts[3].split(':', 1)[1].strip())
    metadata = {
        'date': date,
        'allow_relegation': allow_relegation,
        'allow_sub_training': allow_sub_training,
        'relegation_probability': relegation_probability
    }
    stats = []

    def parse_slash(slash_string):
        """Parses a string with slash notation into a float."""
        try:
            return float(slash_string.split('/')[0].strip())
        except ValueError:
            return 0.0

    for episode in episodes[1:]:
        if episode.strip():
            lines = episode.strip().split('\n')
            if len(lines) > 1:
                report_line = lines[1].strip()
                parts = report_line.split(';')

                episode = int(parse_slash(parts[0].split(':')[1].strip()))
                steps = float(parts[1].split(':')[1].strip())
                succeeded = float(parts[2].split(':')[1].strip())
                score = parse_slash(parts[3].split(':')[1].strip())
                cl = float(parts[4].split(':')[1].strip())
                max_cl = float(parts[5].split(':')[1].strip())
                stats.append({
                    'episode': episode,
                    'steps': steps,
                    'succeeded': succeeded,
                    'score': score,
                    'cl': cl,
                    'max_cl': max_cl 
                })
    return metadata, stats


def compute_moving_average(stats, alpha):
    """Computes the moving average of the scores."""
    output = []
    moving_avg = {}
    for stat in stats:
        for k, v in stat.items():
            moving_avg[k] = moving_avg.get(k, 0.0) * alpha + v * (1 - alpha)
        output.append(moving_avg.copy())
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting TextWorld Plotting Task")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ma_alpha", "-ma", metavar='ma-alpha', type=float, default=0.995, help="Moving average alpha value (default: 0.995)")
    parser.add_argument("--training_trend", "-tt", action='store_true', help="Plot training trend")
    parser.add_argument("--metric", "-m", type=str, default="succeeded", help="Metric to plot (default: succeeded)")
    args = parser.parse_args()

    # Open file dialog to select files
    sessions = list(parse_rollout_file(open_file_dialog(), ma_alpha=args.ma_alpha, training_trend=args.training_trend))
    
    if not sessions:
        logging.error("No files selected or no valid sessions found.")
        sys.exit(1)

    # Plotting the data
    fig, ax = plt.subplots(figsize=(12, 6))
    for session in sessions:
        metadata = session['metadata']
        X = [stat['episode'] for stat in session['stats']]
        Y = [avg[args.metric] for avg in session['moving_average']]
        label = f"{metadata['allow_relegation']} | {metadata['allow_sub_training']} | {metadata['relegation_probability']:.2f}"
        ax.plot(X, Y, label=label)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('TextWorld Rollout Scores')
    ax.legend()
    ax.grid()
    fig.tight_layout()
    plt.show()

    plot_name = f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pgf"
    save_path = save_file_dialog(default_path=plot_name)
    if save_path is not None:
        logging.info(f"Saving plot to {save_path}")
        fig.savefig(save_path, format="pgf", bbox_inches="tight")
        # now clear all the header sections from the file starts with %%
        with open(save_path, 'r') as f:
            lines = f.readlines()
        with open(save_path, 'w') as f:
            for i, line in enumerate(lines):
                if line.startswith('%') and i >= 8:
                    continue
                f.write(line)

    plt.close(fig)