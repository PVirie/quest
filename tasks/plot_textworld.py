import os
import sys
import logging
from datetime import datetime
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import utilities

from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib import lines, markers
from cycler import cycler
import tkinter as tk
from scipy.stats import mannwhitneyu


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


def parse_rollout_file(file_generator, ma_alpha, training_trend, trend_threshold=5000, end_result_threshold=10, filter=False):
    """Parses a rollout file and returns the content."""
    plot_data = {}
    for file_path in file_generator:
        with open(file_path, 'r', encoding="utf-8") as file:
            content = file.read()

        # first detect session prefix "====..."
        for session_text in content.split("========================================================================"):
            session_text = session_text.strip()
            if len(session_text) < 10:
                continue
            metadata_key, metadata, stats = parse_session(session_text)
            last_episode = stats[-1]['episode'] if len(stats) > 0 else 0

            if metadata_key not in plot_data:
                plot_data[metadata_key] = []
            
            if (training_trend and last_episode > trend_threshold) or (not training_trend and last_episode <= end_result_threshold):
                plot_data[metadata_key].append({
                        'file_name': os.path.basename(file_path),
                        'file_path': file_path,
                        'metadata': metadata,
                        'stats': stats,
                    })
            else:
                logging.warning(f"Skipping session in {file_path = } due to {training_trend = } and {last_episode = }")

    # now average paths in the same metadata_key
    for metadata_key, sessions in plot_data.items():
        if len(sessions) == 0:
            logging.warning(f"No valid sessions found for metadata key: {metadata_key}")
            continue
        mean_per_episode, var_per_episode, raw_per_metric = process_stat_sequence([session['stats'] for session in sessions], filter=filter)
        moving_average = compute_moving_average(mean_per_episode, alpha=ma_alpha)
        yield {
            'metadata': sessions[0]['metadata'],
            'stats': mean_per_episode,
            'vars': var_per_episode,
            'moving_average': moving_average,
            'raws': raw_per_metric,
        }


def parse_session(session):
    """
    Date: 2025-06-01 13:45:52
    Allow relegation: True
    Relegation probability: 0.1
    Allow sub training: True
    Allow prospect training: True
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
    # logging.info(f"Parsing session header: {header}")
    field_values = {line.split(':', 1)[0].strip().lower(): line.split(':', 1)[1].strip().lower() for line in header.split('\n')} 
    metadata = {
        'date': utilities.get_datetime_from_string(field_values.get('date', '')),
        'allow_relegation': field_values.get('allow relegation', 'false') == 'true',
        'allow_sub_training': field_values.get('allow sub training', 'false') == 'true',
        'allow_prospect_training': field_values.get('allow prospect training', 'false') == 'true',
        'relegation_probability': float(field_values.get('relegation probability', '0.0'))
    }
    metadata_key = f"{metadata['allow_relegation']}|{metadata['allow_sub_training']}|{metadata['allow_prospect_training']}|{metadata['relegation_probability']:.2f}"
    
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
    stats.sort(key=lambda x: x['episode'])
    return metadata_key, metadata, stats


def compute_moving_average(stats, alpha):
    """Computes the moving average of the scores."""
    output = []
    moving_avg = {}
    for stat in stats:
        for k, v in stat.items():
            moving_avg[k] = moving_avg.get(k, 0.0) * alpha + v * (1 - alpha)
        output.append(moving_avg.copy())
    return output


def process_stat_sequence(session_stat_sequences, filter=False):
    """Computes the mean of a sequence of statistics."""
    if not session_stat_sequences:
        return []

    sums = []
    sqr_sums = []
    raws = {}

    # Iterate over each session's stats
    for session_stats in session_stat_sequences[1:]:
        for i, stat in enumerate(session_stats):
            if i >= len(sums):
                sums.append({})
                sqr_sums.append({})
            filtered = filter and abs(stat.get('succeeded', 0)) < 0.01
            for key, value in stat.items():
                if key not in sums[i]:
                    sums[i][key] = 0.0
                    sqr_sums[i][key] = 0.0
                if key not in raws:
                    raws[key] = []
                if not filtered:
                    sums[i][key] += value
                    sqr_sums[i][key] += value ** 2
                if not filtered or key == 'succeeded':
                    raws[key].append(value)
            sums[i]['count'] = sums[i].get('count', 0) + (1 if not filtered else 0)
            sums[i]['unfiltered_count'] = sums[i].get('unfiltered_count', 0) + 1

    means = []
    vars = []
    for i, stat in enumerate(sums):
        if i >= len(means):
            means.append({})
            vars.append({})
        N = stat.get('count', 1)
        UN = stat.get('unfiltered_count', 1)
        for key in stat:
            if key == 'count':
                continue
            elif key == 'unfiltered_count':
                continue
            elif key == 'succeeded':
                # project the success rate to N
                m = stat[key] / UN if UN > 0 else 0.0
                means[i][key] = m
                vars[i][key] = (sqr_sums[i][key] - UN * m * m) / (UN - 1) if UN > 1 else 0.0
            else:
                m = stat[key] / N if N > 0 else 0.0
                means[i][key] = m
                vars[i][key] = (sqr_sums[i][key] - N * m * m) / (N - 1) if N > 1 else 0.0
        means[i]['count'] = N

    return means, vars, raws


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting TextWorld Plotting Task")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ma-alpha",   "-ma", type=float, default=0.5, help="Moving average alpha value (default: 0.9)")
    parser.add_argument("--end-result", "-end", action='store_true', help="Plot only the end result (default: False)")
    parser.add_argument("--metric",     "-m", type=str, default="score", help="Metric to plot (default: score)")
    parser.add_argument("--filter",     "-f", action='store_true', help="Filter out unsuccessful sessions (default: False)")
    args = parser.parse_args()

    logging.info(f"Arguments: {args}")

    # Open file dialog to select files
    sessions = list(parse_rollout_file(open_file_dialog(), ma_alpha=args.ma_alpha, training_trend=not args.end_result, filter=args.filter))
    
    if not sessions:
        logging.error("No files selected or no valid sessions found.")
        sys.exit(1)

    if args.end_result:
        # print mean and variance of success rate and average context length and max context length group by metadata
        metadata_stats = {}
        for session in sessions:
            metadata = session['metadata']
            label = f"{metadata['allow_relegation']} | {metadata['allow_sub_training']} | {metadata['allow_prospect_training']} | {metadata['relegation_probability']:.2f}"
            if label not in metadata_stats:
                metadata_stats[label] = {
                    'count': 0,
                    'steps': 0, 'succeeded': 0, 'score': 0, 'cl': 0, 'max_cl': 0,  
                    'step_2': 0, 'succeeded_2': 0, 'score_2': 0, 'cl_2': 0, 'max_cl_2': 0,
                }
            for means, vars in zip(session['stats'], session['vars']):
                N = means.get('count', 1)
                metadata_stats[label]['count'] += N
                metadata_stats[label]['steps'] += means['steps'] * N
                metadata_stats[label]['step_2'] += vars['steps'] * (N - 1) + means['steps'] ** 2 * N
                metadata_stats[label]['succeeded'] += means['succeeded'] * N
                metadata_stats[label]['succeeded_2'] += vars['succeeded'] * (N - 1) + means['succeeded'] ** 2 * N
                metadata_stats[label]['score'] += means['score'] * N
                metadata_stats[label]['score_2'] += vars['score'] * (N - 1) + means['score'] ** 2 * N
                metadata_stats[label]['cl'] += means['cl'] * N
                metadata_stats[label]['cl_2'] += vars['cl'] * (N - 1) + means['cl'] ** 2 * N
                metadata_stats[label]['max_cl'] += means['max_cl'] * N
                metadata_stats[label]['max_cl_2'] += vars['max_cl'] * (N - 1) + means['max_cl'] ** 2 * N

        # print the results
        logging.info("End Result Statistics:")
        for label, stats in metadata_stats.items():
            N = stats['count']
            if N > 0:
                avg_steps = stats['steps'] / N
                var_steps = (stats['step_2'] - N * (avg_steps ** 2)) / (N - 1)
                avg_succeeded = stats['succeeded'] / N
                var_succeeded = (stats['succeeded_2'] - N * (avg_succeeded ** 2)) / (N - 1)
                avg_score = stats['score'] / N
                var_score = (stats['score_2'] - N * (avg_score ** 2)) / (N - 1)
                avg_cl = stats['cl'] / N
                var_cl = (stats['cl_2'] - N * (avg_cl ** 2)) / (N - 1)
                avg_max_cl = stats['max_cl'] / N
                var_max_cl = (stats['max_cl_2'] - N * (avg_max_cl ** 2)) / (N - 1)
                logging.info(f"{label} - Count: {N}")
                logging.info(f"{label} - Avg Steps: {avg_steps:.2f}, Avg Succeeded: {avg_succeeded:.2f}, Avg Score: {avg_score:.2f}, Avg CL: {avg_cl:.2f}, Avg Max CL: {avg_max_cl:.2f}")
                logging.info(f"{label} - Std Steps: {var_steps ** 0.5:.2f}, Std Succeeded: {var_succeeded ** 0.5:.2f}, Std Score: {var_score ** 0.5:.2f}, Std CL: {var_cl ** 0.5:.2f}, Std Max CL: {var_max_cl ** 0.5:.2f}")
                # Print LaTeX formatted output
                logging.info(f"{avg_score:.2f} $\\pm$ {var_score ** 0.5:.2f} & {avg_succeeded:.2f} $\\pm$ {var_succeeded ** 0.5:.2f} & {avg_cl:.2f} $\\pm$ {var_cl ** 0.5:.2f} \\\\")
            else:
                logging.warning(f"{label} - No valid data found.")

        logging.info("=========================================================")
        logging.info("Mann-Whitney U Test Results:")

        # calculate the Mann-Whitney U test for each pair of metadata
        stat_p_value_table = [[[(0.0, 0.0) for _ in range(len(sessions))] for _ in range(len(sessions))] for metric in ['succeeded', 'cl', 'max_cl']]
        for i, session_i in enumerate(sessions):
            for j, session_j in enumerate(sessions):
                if i == j:
                    continue
                metadata_i = session_i['metadata']
                metadata_j = session_j['metadata']
                label = f"{metadata_i['allow_relegation']} | {metadata_i['allow_sub_training']} | {metadata_i['allow_prospect_training']} | {metadata_i['relegation_probability']:.2f}"
                label2 = f"{metadata_j['allow_relegation']} | {metadata_j['allow_sub_training']} | {metadata_j['allow_prospect_training']} | {metadata_j['relegation_probability']:.2f}"
                raws_i = session_i['raws']
                raws_j = session_j['raws']

                for k, metric in enumerate(['score', 'succeeded', 'cl']):
                    # Mann-Whitney U test
                    if metric == 'succeeded':
                        alternative = 'greater'
                    elif metric == 'cl':
                        alternative = 'less'
                    elif metric == 'score':
                        alternative = 'greater'
                    else:
                        alternative = 'two-sided'
                    u_stat, p_value = mannwhitneyu(
                        raws_i[metric],
                        raws_j[metric],
                        alternative=alternative
                    )
                    stat_p_value_table[k][i][j] = (u_stat, p_value)
                    logging.info(f"Mann-Whitney U test for {alternative} of {metric} between {label} and {label2}: U={u_stat:.2f}, p={p_value:.4f}")
            logging.info("------------------------------------------------")

    else:
        # Plotting the data
        style_cycler = cycler(
            color=plt.cm.tab10.colors,
            linestyle=['-', '-.', ':', '-', '-.', ':', '-', '-.', ':', '-'],
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_prop_cycle(style_cycler)
        for n, session in enumerate(sessions):
            metadata = session['metadata']
            X = [stat['episode'] for stat in session['stats']]
            Y = [avg[args.metric] for avg in session['moving_average']]
            STD = [var[args.metric] ** 0.5 for var in session['vars']]
            label = f"{metadata['allow_relegation']} | {metadata['allow_sub_training']} | {metadata['allow_prospect_training']} | {metadata['relegation_probability']:.2f}"
            # ax.fill_between(X, [Y[i] - STD[i] for i in range(len(Y))], [Y[i] + STD[i] for i in range(len(Y))], alpha=0.1)
            ax.errorbar(X[n::5], Y[n::5], yerr=STD[n::5], markersize=3, capsize=3, label=label, elinewidth=1, markeredgewidth=1, linewidth=2)
        ax.set_xlabel('episode')
        ax.set_ylabel(args.metric)
        ax.set_title('TextWorld Rollout Scores')
        # ax.legend()
        # ax.grid()
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