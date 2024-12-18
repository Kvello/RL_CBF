import pandas as pd 
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from tqdm import tqdm

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],  # Optional: Customize to match LaTeX fonts
    "axes.labelsize": 12,  # Label font size
    "font.size": 10,  # General font size
    "legend.fontsize": 10,  # Legend font size
    "xtick.labelsize": 10,  # x-axis tick label size
    "ytick.labelsize": 10,  # y-axis tick label size
})

def plot_collisions(collision_file_sets:Dict[str, Dict[str,Path]],
                    end_steps, 
                    window_size,
                    column_name:str,
                    x_label:str,
                    y_label:str,
                    out_filename:str):

    """
    Calculate the average number of collisions until end_steps
    averaged over each set of files. Calculate also the standard deviation.
    Present the results in a bar plot.

    Args:
        collision_file_sets (List[List[str]]): List of sets of files
        end_steps_sets (List[int]): List of end_steps for each file
    """
    set_names = list(collision_file_sets.keys())
    sets_stats = {
        x_label: [],
        y_label: []
    }
    for set_name, file_set in collision_file_sets.items():
        for file_num, tuple_ in enumerate(file_set.items()):
            file_name, file_path = tuple_
            df = pd.read_csv(file_path)
            end_step = end_steps.get(file_name, None)
            if end_step is not None:
                collisions = df[df['Step'] <= end_step][file_name + column_name]
            else:
                print(f'End step not found for file {file_name}')
                collisions = df[file_name + column_name]
            if column_name == ' - step_crash':
                # The first step is allways a collision, so ignore
                collisions = collisions[1:]
            sets_stats[x_label].append(set_name)
            sets_stats[y_label].append(np.sum(collisions).astype(float)*window_size)
            # print(f"Set: {set_name}, File: {file_name}, Training Collisions: {np.sum(collisions)*window_size}")

    collision_data = pd.DataFrame({
        x_label : sets_stats[x_label],
        "Number of Collisions": sets_stats[y_label]
    })
    if len(collision_data) > len(set(collision_data[x_label])):
        # We have multiple entries for the same method - use boxplot
        sns.boxplot(data = collision_data, x = x_label, y = "Number of Collisions",
                    hue=x_label,palette="pastel", whis=[0,100])
    sns.stripplot(data = collision_data, x = x_label, y = "Number of Collisions",
                  size=10, color=".3")
    plt.title(y_label)
    # Get todays date and time, and add to the filename for uniqueness
    
    datestr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = out_filename + datestr + ".pdf"
    plt.savefig(filename)
    plt.close()

def plot_timeseries_step(data_sets:Dict[str,Dict[str,Path]],
                         end_steps,
                         column_name:str,
                         hue_label:str,
                         y_label:str,
                         out_filename:str,
                         log_y:bool = False):
    
    """
    Plot the time series of the given data sets.
    """
    data_frames = []
    for set_name, file_set in data_sets.items():
        for file_num, tuple_ in enumerate(file_set.items()):
            file_name, file_path = tuple_
            df = pd.read_csv(file_path)
            end_step = end_steps.get(file_name, None)
            if end_step is not None:
                df = df[df['Step'] <= end_step][['Step', file_name + column_name]]
            else:
                print(f'End step not found for file {file_name}')
                df = df[['Step', file_name + column_name]]
            df = df.rename(columns={file_name + column_name: y_label})
            # Interpolate over missing steps
            df = df.set_index('Step').reindex(
                range(0, df['Step'].max() + 1)
                ).interpolate(method='index').reset_index()
            df['Step'] = df['Step']*0.5
            df[hue_label] = set_name
            df["size"] = 1/(1+2*abs(df[y_label].mean()))
            data_frames.append(df)
    complete_df = pd.concat(data_frames,ignore_index=True)
    # Set to display all rows
    ax = sns.lineplot(data=complete_df, x='Step', y=y_label, hue=hue_label,palette="pastel",
                 style=hue_label, markers=False, dashes=False,errorbar=('ci',95),
                 alpha=0.99, size="size", sizes=(1, 3),legend=True)
    handles, labels = ax.get_legend_handles_labels()
    legend_title = ax.get_legend().get_title().get_text()
    filtered_handles_labels = [
        (h, l) for h, l in zip(handles, labels) if l in complete_df[hue_label].unique()
    ]
    if filtered_handles_labels:
        filtered_handles, filtered_labels = zip(*filtered_handles_labels)
        ax.legend(filtered_handles, filtered_labels, title="$\mu$")
    else:
        ax.legend([], [])  # Remove legend entirely if no relevant labels

    if log_y:
        plt.yscale('log')
    datestr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = out_filename + datestr + ".pdf"
    plt.grid()
    plt.savefig(filename)
    plt.close()
    
    
    
    
    
def plot_timeseries(data_sets:Dict[str,Dict[str,Path]],
                    epochs_sets: Dict[str,Dict[str,Path]],
                    end_steps,
                    column_name:str, 
                    hue_label:str, 
                    y_label:str,
                    out_filename:str,
                    log_y:bool = False):
    """
    Plot the time series of the given data sets.

    Args:
        data_sets (List[List[str]]): List of sets of files
        end_steps_sets (List[int]): List of end_steps for each file
    """
    epoch_files_merged = {k:v for s in epochs_sets.values() for k,v in s.items()}
    data_frames = []
    for set_name, file_set in data_sets.items():
        for file_num, tuple_ in enumerate(file_set.items()):
            file_name, file_path = tuple_
            df = pd.read_csv(file_path)
            end_step = end_steps.get(file_name, None)
            if end_step is not None:
                df = df[df['Step'] <= end_step][['Step', file_name + column_name]]
            else:
                print(f'End step not found for file {file_name}')
            df = df.rename(columns={file_name + column_name: y_label})
            # Interpolate over missing steps
            df = df.set_index('Step').reindex(
                range(0, df['Step'].max() + 1)
                ).interpolate(method='index').reset_index()
            epoch_df = pd.read_csv(epoch_files_merged[file_name])
            epoch_df = epoch_df.set_index('Step').reindex(
                range(0, df['Step'].max() + 1,)
                ).interpolate(method='index',extrapolate="both").reset_index()
            df['Epoch'] = epoch_df[file_name + ' - info/epochs']
            df = df.drop_duplicates(subset='Epoch', keep='first')
            df.dropna(inplace=True)
            df = df.set_index('Epoch').reindex(
                range(0, 250)
                ).interpolate(method='index').reset_index()
            df[hue_label] = set_name
            df['file'] = file_name
            data_frames.append(df)
    complete_df = pd.concat(data_frames,ignore_index=True)
    # Set to display all rows
    sns.lineplot(data=complete_df, x='Epoch', y=y_label, hue=hue_label,palette="pastel",
                 style=hue_label, markers=False, dashes=False,errorbar=('ci',95),alpha=0.99)
    if log_y:
        plt.yscale('log')
    datestr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = out_filename + datestr + ".pdf"
    plt.savefig(filename)
    plt.close()

def plot_training_steps(step_file_sets:Dict[str,Dict[str,Path]], end_steps):
    """
    Calculate the average number of steps until end_steps
    averaged over each set of files. Calculate also the standard deviation.
    Present the results in a bar plot.

    Args:
        step_file_sets (List[List[str]]): List of sets of files
        end_steps_sets (List[int]): List of end_steps for each file
    """
    set_names = list(step_file_sets.keys())
    sets_stats = {
        "Method": [],
        "Training Steps": []
    }
    for set_name, file_set in step_file_sets.items():
        for file_num, tuple_ in enumerate(file_set.items()):
            file_name, file_path = tuple_
            df = pd.read_csv(file_path)
            end_step = end_steps.get(file_name, None)
            if end_step is not None:
                # Find first step where the step is greater or equal to end_step
                steps = df[df['Step'] <= end_step][file_name + ' - global_step']
                total_steps = steps.iloc[-1]
            else:
                print(f'End step not found for file {file_name}')
                steps = df[file_name + ' - global_step']
                total_steps = steps.iloc[-1]
            sets_stats["Method"].append(set_name)
            sets_stats["Training Steps"].append(total_steps)
            print(f"Set: {set_name}, File: {file_name}, Training Steps: {total_steps}")

    step_data = pd.DataFrame({
        "Method": sets_stats["Method"],
        "Number of Steps": sets_stats["Training Steps"]
    })
    sns.boxplot(data = step_data, x = "Method", y = "Number of Steps",
                hue="Method",palette="pastel", whis=[0,100])
    sns.stripplot(data = step_data, x = "Method", y = "Number of Steps",
                  size=10, color=".3")
    plt.title("Training Steps")
    # Get todays date and time, and add to the filename for uniqueness
    datestr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = "training_steps" + datestr + ".pdf"
    plt.savefig(filename)
    plt.close()

def plot_training_epochs(epoch_file_sets:Dict[str,Dict[str,Path]], end_steps):
    """
    Calculate the average number of epochs until end_steps
    averaged over each set of files. Calculate also the standard deviation.
    Present the results in a bar plot.

    Args:
        epoch_file_sets (List[List[str]]): List of sets of files
        end_steps_sets (List[int]): List of end_steps for each file
    """
    set_names = list(epoch_file_sets.keys())
    sets_stats = {
        "Method": [],
        "Training Epochs": []
    }
    for set_name, file_set in epoch_file_sets.items():
        for file_num, tuple_ in enumerate(file_set.items()):
            file_name, file_path = tuple_
            df = pd.read_csv(file_path)
            end_step = end_steps.get(file_name, None)
            if end_step is not None:
                # Find first step where the step is greater or equal to end_step
                epochs = df[df['Step'] <= end_step][file_name + ' - info/epochs']
                total_epochs = epochs.iloc[-1]
            else:
                print(f'End step not found for file {file_name}')
                epochs = df[file_name + ' - Epoch']
                total_epochs = epochs.iloc[-1]
            sets_stats["Method"].append(set_name)
            sets_stats["Training Epochs"].append(total_epochs)
            print(f"Set: {set_name}, File: {file_name}, Training Epochs: {total_epochs}")

    epoch_data = pd.DataFrame({
        "Method": sets_stats["Method"],
        "Number of Epochs": sets_stats["Training Epochs"]
    })
    sns.boxplot(data = epoch_data, x = "Method", y = "Number of Epochs",
                hue="Method",palette="pastel", whis=[0,100])
    sns.stripplot(data = epoch_data, x = "Method", y = "Number of Epochs",
                  size=10, color=".3")
    plt.title("Training Epochs")
    # Get todays date and time, and add to the filename for uniqueness
    datestr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = "training_epochs" + datestr + ".pdf"
    plt.savefig(filename)
    plt.close() 

def find_csv_files(root_dirs:List[str],set_names=List[str]) -> Dict[str,Dict[str,Path]]:
    """
    Recursively search for all CSV files starting from the given root directory.
    
    :param root_dir: Root directory to start the search
    :return: Dictionary of file names and their corresponding Path objects
    """
    # Create a Path object for the directory
    sets = {name:{} for name in set_names}
    for name,root_dir in zip(set_names,root_dirs):
        root_path = Path(root_dir)
        sets[name] = {file.name.split('.')[0] :file for file in root_path.rglob('*.csv')}
    # Use rglob() to find all files with .csv extension recursively
    return sets

def find_steps_to_successrate(successrate_files:Dict[str,Dict[str,Path]], 
                              epoch_files:Dict[str,Dict[str,Path]],
                              epoch_thresold,
                              threshold:float)->Dict[str,int]:
    """
    Find the step number where the successrate exceeds the given threshold for each file.
    
    :param successrate_files: Dictionary of file names and their corresponding Path objects
    :param threshold: Successrate threshold
    :return: Dictionary of file names and their corresponding step numbers
    """
    # Initialize an empty dictionary to store the step number for each file
    end_steps = {}
    successrate_files_merged = {k:v for s in successrate_files.values() for k,v in s.items()} 
    epoch_files_merged = {k:v for s in epoch_files.values() for k,v in s.items()}
    # Iterate over each file
    for name, file in successrate_files_merged.items():
        # Read the CSV file
        epoch_file = epoch_files_merged[name] 
        epoch_df = pd.read_csv(epoch_file)
        excessive_epochs = epoch_df[epoch_df[name + ' - info/epochs'] >= epoch_thresold]
        if not excessive_epochs.empty:
            end_steps[name] = int(min(excessive_epochs['Step']))
        else:
            end_steps[name] = int(max(epoch_df['Step']))
        df = pd.read_csv(file)
        # Find the step number where the successrate exceeds the threshold
        filtered_df = df[df[name + ' - Success Rate'] >= threshold]
        if not filtered_df.empty:
            first_successrate_occurence = int(min(filtered_df['Step']))
            end_steps[name] = min(first_successrate_occurence, end_steps[name])
        else:
            print("Successrate threshold not reached for file: ", name)
    return end_steps

def main(paths,set_names, has_cbf, plot_type):
    # Find all csv files in the specified path
    file_types = ['successrate', 'rewards', 'cbfconstraint', 'cbfvalue', 'collisions',
                  'crashrate','step','epochs']
    csv_file_sets = find_csv_files(paths,set_names)
    sets = {t:{name:{} for name in set_names} for t in file_types}
    for set_name,file_set in csv_file_sets.items():
        sets["successrate"][set_name] = {"-".join(name.split('-')[1:]):file for name, file in 
                         file_set.items() if 'successrate'==name.split('-')[0]}

        sets["rewards"][set_name] ={"-".join(name.split('-')[1:]):file for name, file in 
                        file_set.items() if 'rewards'==name.split('-')[0]}

        sets["cbfconstraint"][set_name] = {"-".join(name.split('-')[1:]):file for name, file 
                            in file_set.items() if 'cbfconstraint'==name.split('-')[0]}

        sets["cbfvalue"][set_name] = {"-".join(name.split('-')[1:]):file for name, file in
                        file_set.items() if 'cbfvalue'==name.split('-')[0]}

        sets["collisions"][set_name] = {"-".join(name.split('-')[1:]):file for name, file in 
                            file_set.items() if 'collisions'==name.split('-')[0]}
        sets["crashrate"][set_name] = {"-".join(name.split('-')[1:]):file for name, file in
                            file_set.items() if 'crashrate'==name.split('-')[0]}
        sets["step"][set_name] = {"-".join(name.split('-')[1:]):file for name, file in
                            file_set.items() if 'step'==name.split('-')[0]}
        sets["epochs"][set_name] = {"-".join(name.split('-')[1:]):file for name, file in
                            file_set.items() if 'epochs'==name.split('-')[0]}
        
    successrate_threshold = 0.95
    epoch_thresold = 250
    window_size = 2050 # Number of episodes the rate files are averaged over
    # Find the step number where the successrate exceeds the threshold for each file
    end_steps = find_steps_to_successrate(sets["successrate"],
                                          sets["epochs"],
                                          epoch_thresold, 
                                          successrate_threshold)
    if 'training-collisions' in plot_type or plot_type == 'all':
        plot_collisions(sets["crashrate"], end_steps,window_size, 
                        "Method", 
                        "Training Collisions", 
                        "training_collisions")
    if 'steps' in plot_type or plot_type == 'all':
        plot_training_steps(sets["step"], end_steps)
    if 'epochs' in plot_type or plot_type == 'all':
        plot_training_epochs(sets["epochs"], end_steps)
    if 'crashrate-timeseries' in plot_type or plot_type == 'all':
        plot_timeseries(sets["crashrate"],
                        sets["epochs"],
                        end_steps,
                        " - Crash Rate",
                        "Method",
                        "Crash Rate",
                        "crashrate_timeseries")
    if 'successrate-timeseries' in plot_type or plot_type == 'all':
        plot_timeseries(sets["successrate"],
                        sets["epochs"],
                        end_steps,
                        " - Success Rate",
                        "Method", 
                        "Success Rate", 
                        "successrate_timeseries")
    if 'cbf-constraint' in plot_type or plot_type == 'all':
        plot_timeseries_step(sets["cbfconstraint"],
                        end_steps,
                        " - Smallest CBF constraint(unfiltered)",
                        "$\mu$", 
                        "CBF Constraint", 
                        "cbf_constraint_timeseries",
                        log_y = False)
    if 'cbf-values' in plot_type or plot_type == 'all':
        plot_timeseries(sets["cbfvalue"],
                        sets["step"],
                        end_steps,
                        " - Smallest CBF",
                        "Method", 
                        "CBF Value", 
                        "cbf_value_timeseries",
                        log_y = True)
    if 'final-collisions' in plot_type or plot_type == 'all':
        plot_collisions(sets["collisions"],
                        end_steps,
                        1,
                        " - step_crash",
                        "Method",
                        "Collisions",
                        "eval_collisions")
if __name__ == '__main__':
    # Take command line arguments specifying search path for csv files
    parser = argparse.ArgumentParser(description='Plot data from csv files')
    parser.add_argument('--paths', type=str,nargs='+',
                        help='Paths to sets of csv files')
    parser.add_argument('--set-names', type=str,nargs='+',
                        help='Names of the sets of csv files')
    parser.add_argument('--has_cbf', type=bool, help='Whether the csv files have a colum for \
                        cbf values, and cbf constraint values', default=False)
    parser.add_argument('--plot-type', type=str, choices=['training-collisions', 
                                                          'steps', 
                                                          'epochs',
                                                          'final-collisions',
                                                          'cbf-constraint',
                                                          'cbf-values',
                                                          'successrate-timeseries',
                                                          'crashrate-timeseries',
                                                          'all'], 
                        default='all',nargs='+')
    
    args = parser.parse_args()
    assert len(args.paths) == len(args.set_names), "Number of paths and set names must be equal"
    sns.set_theme(style="ticks")
    main(args.paths,args.set_names, args.has_cbf, args.plot_type)
