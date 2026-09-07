
# Libraries 
import numpy as np
import matplotlib.pyplot as plt



def plot_global_importance(attributions, neuron_names, top_n=10, title="Global Neuron Importance"):
    """
    Bar plot for top n neurons across all time window - can be used per behaviour or per all behaviours combined

    Arguments:
    attributions = output of a xai method function
    nauron_names = names of neurons used in deep learning model 
    top_n = number of top neurons shown on the graph
    title = title of the plot

    """

    mean_attr = np.mean(np.abs(attributions), axis=(0, 1))
    sorted_idx = np.argsort(mean_attr)[-top_n:]
    sorted_idx = sorted_idx[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(np.array(neuron_names)[sorted_idx], mean_attr[sorted_idx], color='#B03060')
    plt.title(title)
    plt.xlabel("Mean Absolute IG Score")
    plt.ylabel("Neuron")
    plt.tight_layout()
    plt.show()


def plot_transition_heatmap(attributions_window, neuron_names, top_n=15, title="Transition Trigger Heatmap"):
    
    """
    Generuje mapę cieplną dla pojedynczego okna czasowego (np. przy zmianie zachowania).
    """

    mean_attr = np.mean(np.abs(attributions_window), axis=1) 
    sorted_idx = np.argsort(mean_attr)[-top_n:]
    
    filtered_attr = attributions_window[sorted_idx, :]
    filtered_names = np.array(neuron_names)[sorted_idx]
    
    plt.figure(figsize=(12, 6))
    im = plt.imshow(filtered_attr, aspect='auto', cmap='coolwarm', 
                    vmin=-np.max(np.abs(filtered_attr)), vmax=np.max(np.abs(filtered_attr)))
    
    plt.yticks(ticks=np.arange(top_n), labels=filtered_names)
    plt.xlabel("Time steps (frames within window)")
    plt.title(title)
    plt.colorbar(im, label="IG Attribution Score")
    plt.tight_layout()
    plt.show()