import os
import sys
from tkinter import filedialog
import logging

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
            date, steps = parse_session(session_text)
            yield {
                'file_path': file_path,
                'date': date,
                'steps': steps,
            }


def parse_session(session):
    """
    Date: 2025-06-01 13:45:52
    ------------------------------------------------------------------------
    Episode 10
    [Report]	episode: 10/20000; steps: 100.0; score: -2.0/ 1.2; cl: 100.0; max cl: 100.0
    ...
    ------------------------------------------------------------------------
    Episode 20
    ...

    """
    episodes = session.strip().split('------------------------------------------------------------------------')
    header = episodes[0].strip()
    logging.info(f"Parsing session header: {header}")
    date = utilities.get_datetime_from_string(header.split("Date: ")[-1].strip())
    steps = []

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
                # handle backward if "steps" is in part[0]
                if "steps" in parts[0]:
                    sub_parts = parts[0].split('steps')
                    parts = [sub_parts[0]] + ['steps' + sub_parts[1]] + parts[1:]
                step_info = {
                    'episode': int(parse_slash(parts[0].split(':')[1].strip())),
                    'steps': float(parts[1].split(':')[1].strip()),
                    'score': parse_slash(parts[2].split(':')[1].strip()),
                    'cl': float(parts[3].split(':')[1].strip()),
                    'max_cl': float(parts[4].split(':')[1].strip())
                }
                steps.append(step_info)
    return date, steps



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting TextWorld Plotting Task")

    # Open file dialog to select files
    sessions = list(parse_rollout_file(open_file_dialog()))
    
    if not sessions:
        logging.error("No files selected or no valid sessions found.")
        sys.exit(1)

    # Plotting the data
    plt.figure(figsize=(12, 6))
    for session in sessions:
        date = session['date']
        steps = session['steps']
        episodes = [step['episode'] for step in steps]
        scores = [step['score'] for step in steps]
        plt.plot(episodes, scores, label=date.strftime("%Y-%m-%d %H:%M:%S"))
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('TextWorld Rollout Scores')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()