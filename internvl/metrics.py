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
