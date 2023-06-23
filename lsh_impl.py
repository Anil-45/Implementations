"""Locality Sensitive Hashing implementation using cosine similarity."""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

N_HYPERPLANES = 5
K_NEIGHBOURS = 5

def generate_hyperplanes(size: int) -> np.array:
    """Generates Hyperplanes.

    Args:
        size (int): Dimension of hyperplane

    Returns:
        np.array: Hyperplanes
    """
    random_hyperplanes = []
    for i in range(N_HYPERPLANES):
        random_hyperplanes.append(np.random.normal(0, 1, size))
    return np.array(random_hyperplanes)

def generate_bins() -> dict:
    """Generate Hashtable.

    Returns:
        dict: Hashtable
    """
    bins = dict()
    for i in range(2**N_HYPERPLANES):
        bins[i] = []
    return bins

def fill_bins(
    data: np.array,
    tar: np.array,
    bins: dict,
    hyperplanes: np.array
) -> None:
    """Fill hashtable.

    Args:
        data (np.array): Training data
        tar (np.array): Training labels
        bins (dict): Hashtable to fill
        hyperplanes (np.array): Hyperplanes
    """
    for i in range(data.shape[0]):
        hash = 0
        for j in range(hyperplanes.shape[0]):
            if np.dot(hyperplanes[j], data[i, :]) >= 0.0:
                hash += 2**j
        
        bins[hash].append((data[i, :], tar[i]))

def predict(data: np.array, bins: dict, hyperplanes: np.array) -> np.array:
    """Predict.

    Args:
        data (np.array): Test data
        bins (dict): Hashtable
        hyperplanes (np.array): Hyperplanes

    Returns:
        np.array: Predictions
    """
    predictions = []
    for i in range(data.shape[0]):
        hash = 0
        for j in range(hyperplanes.shape[0]):
            if np.dot(hyperplanes[j], data[i, :]) >= 0.0:
                hash += 2**j
        
        neighbours = list(bins[hash])
        if 0 == len(neighbours):
            predictions.append(-1)
            continue
        
        cosine_sim = []
        for j in range(len(neighbours)):
            ntr = np.dot(neighbours[j][0], data[i, :])
            dtr = (np.linalg.norm(neighbours[j][0])*np.linalg.norm(data[i, :]))
            cosine_sim.append(ntr/dtr)
        
        top_k_neighbours = np.argsort(cosine_sim)[::-1][:K_NEIGHBOURS]
        class_of_neighbours = [neighbours[n][1] for n in top_k_neighbours]
        class_of_most_neigh = np.argmax(np.bincount(class_of_neighbours))  
        predictions.append(class_of_most_neigh)
    return np.array(predictions)

if "__main__" == __name__:    
    X, y = load_iris(return_X_y=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        stratify=y,
                                                        shuffle=True,
                                                        random_state=0)
    
    hyperplanes = generate_hyperplanes(X_train.shape[1])
    bins = generate_bins()
    fill_bins(X_train, y_train, bins, hyperplanes)
    predictions = predict(X_test, bins, hyperplanes)
    print(accuracy_score(y_test, predictions))
    