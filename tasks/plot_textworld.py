import os
import sys
from tkinter import filedialog
import logging
from datetime import datetime

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


def parse_rollout_file(file_generator):
    """Parses a rollout file and returns the content."""
    for file_path in file_generator:
        with open(file_path, 'r', encoding="utf-8") as file:
            content = file.read()

        # first detect session prefix "====..."
        for session_text in content.split("========================================================================"):
            session_text = session_text.strip()
            if len(session_text) < 10:
                continue
            date, stats = parse_session(session_text)
            yield {
                'file_path': file_path,
                'date': date,
                'stats': stats,
                'moving_average': compute_moving_average(stats)
            }


def parse_session(session):
    """
    Date: 2025-06-01 13:45:52
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
    date = utilities.get_datetime_from_string(header.split("Date: ")[-1].strip())
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
    return date, stats


def compute_moving_average(stats, alpha=0.995):
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

    # Open file dialog to select files
    sessions = list(parse_rollout_file(open_file_dialog()))
    
    if not sessions:
        logging.error("No files selected or no valid sessions found.")
        sys.exit(1)

    # Plotting the data
    fig, ax = plt.subplots(figsize=(12, 6))
    for session in sessions:
        date = session['date']
        X = [stat['episode'] for stat in session['stats']]
        Y = [avg['succeeded'] for avg in session['moving_average']]
        ax.plot(X, Y, label=date.strftime("%Y-%m-%d %H:%M:%S"))
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