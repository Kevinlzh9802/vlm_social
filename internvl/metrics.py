import numpy as np
from collections import defaultdict

def get_all_elems(x: list):
    return list(set(element for sublist1 in x for sublist2 in sublist1 for element in sublist2))

def compute_HIC(T, P, G, GT):
    # Determine the maximum cardinalities in detected and ground-truth groups
    max_gt_cardinality = max(max(len(gt) for gt in GT[t]) for t in T)
    max_detected_cardinality = max(0 if not G[t] else max(len(g) for g in G[t]) for t in T)

    # Initialize the HIC matrix
    HIC = np.zeros((max_gt_cardinality, max_detected_cardinality))

    # Calculate HIC(i, j) according to the formula
    for t in T:
        g_t = G[t]
        gt_t = GT[t]
        if not g_t or not gt_t:
            continue

        for i in range(1, max_gt_cardinality + 1):
            for j in range(1, max_detected_cardinality + 1):
                a = any(len(group) == j for group in g_t)
                b = any(len(group) == i for group in gt_t)
                if a and b:
                    for p in P:
                        if dij(p, G[t], GT[t], i, j):
                            HIC[i - 1][j - 1] += 1

    # Normalize each row by the number of people in ground-truth groups of cardinality i
    for i in range(max_gt_cardinality):
        ni = sum(len(gt) for t in T for gt in GT[t] if len(gt) == i + 1)
        if ni > 0:
            HIC[i] /= ni

    return HIC


def dij(p, G_t, GT_t, i, j):
    # Check if person p is in a detected group of cardinality j and a ground-truth group of cardinality i
    in_detected_group = any(p in group and len(group) == j for group in G_t)
    in_ground_truth_group = any(p in gt_group and len(gt_group) == i for gt_group in GT_t)
    return 1 if in_detected_group and in_ground_truth_group else 0


def HIC_stats(HIC):
    """
    Calculate various metrics from the given HIC matrix.

    Args:
        HIC (numpy.ndarray): The HIC confusion matrix.

    Returns:
        dict: A dictionary containing all calculated metrics.
    """
    # Matrix dimensions
    N = HIC.shape[0]

    # Cardinality level accuracy (A)
    # A = np.sum(np.diag(HIC)) / np.sum(HIC)

    # Cardinality precision, recall, and F1 score for each class
    precision = []
    recall = []
    f1_scores = []
    for C in range(N):
        Pr = HIC[C, C] / np.sum(HIC[:, C]) if np.sum(HIC[:, C]) > 0 else 0
        Re = HIC[C, C] / np.sum(HIC[C, :]) if np.sum(HIC[C, :]) > 0 else 0
        F1 = (2 * Pr * Re) / (Pr + Re) if Pr + Re > 0 else 0
        precision.append(Pr)
        recall.append(Re)
        f1_scores.append(F1)

    # # Cardinality deviation (D)
    # diagonal = np.diag(HIC)
    # mu = np.mean(diagonal)
    # D = np.sqrt(np.sum((diagonal - mu) ** 2) / N)
    #
    # # Upper-lower difference (UL)
    # upper_sum = np.sum(HIC[np.triu_indices(N, k=1)])  # Sum of upper triangular elements
    # lower_sum = np.sum(HIC[np.tril_indices(N, k=-1)])  # Sum of lower triangular elements
    # UL = upper_sum - lower_sum
    #
    # # Weighted upper-lower difference (WUL)
    # WUL_upper = np.sum([(j - i) * HIC[i, j] for i in range(N) for j in range(i + 1, N)])
    # WUL_lower = np.sum([(i - j) * HIC[j, i] for i in range(N) for j in range(i + 1, N)])
    # WUL = WUL_upper - WUL_lower

    return {
        # "Accuracy (A)": A,
        "precision": np.array(precision).reshape(1, -1),
        "recall": np.array(recall).reshape(1, -1),
        "f1": np.array(f1_scores).reshape(1, -1),
        # "Cardinality Deviation (D)": D,
        # "Upper-Lower Difference (UL)": UL,
        # "Weighted Upper-Lower Difference (WUL)": WUL,
    }


def main():
    # Example time instants (frames)
    T = [1, 2]

    # List of detected people across all frames
    P = ['A', 'B', 'C', 'D']

    # Detected groups per frame
    G = {
        1: [['A', 'B'], ['C']],
        2: [['A', 'C'], ['D']]
    }

    # Ground-truth groups per frame
    GT = {
        1: [['A'], ['B', 'C']],
        2: [['A', 'C'], ['D']]
    }

    # Compute the HIC matrix
    HIC_matrix = compute_HIC(T, P, G, GT)
    print("HIC Matrix:")
    print(HIC_matrix)

if __name__ == '__main__':
    main()
