from math import radians
from math import cos

import math
import random
import numpy as np
from collections import defaultdict
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import time
from heapq import heappush, heappop

from datetime import datetime

# Configuration parameters
NUM_USERS = 1000
MAX_RECORDS_PER_USER = 25
CHECKINS_FILE = '../loc-gowalla_totalCheckins.txt'
EDGES_FILE = '../loc-gowalla_edges.txt'
LAT_RANGE = (37.735, 37.81151)
LONG_RANGE = (-122.5, -122.38)
RADIUS = 1000
EPS_VALUES = [0.5, 1.0, 2.0, 3.0]
#EPS_VALUES = [1.0]
DELAY_MAX_VALUES = [5, 10, 15, 20]
NUM_CENTERS = 10
NUM_ROUNDS = 10
METHODS = ["SMC1", "SMC2", "SDP1", "SDP2", "CDP"]
DISTRIBUTIONS = {
    "uniform": {method: 1 / len(METHODS) for method in METHODS},
    "custom1": {"SMC1": 0.35, "SMC2": 0.35, "SDP1": 0.125, "SDP2": 0.125, "CDP": 0.05},
    "custom2": {"SMC1": 0.125, "SMC2": 0.125, "SDP1": 0.35, "SDP2": 0.35, "CDP": 0.05}
}

USER_COUNTS = [500, 1000, 1500, 2000]  # Numbers of users to test
DEFAULT_EPS = 1.0  # Default when testing other variables
DEFAULT_USERS = 1000  # Default when testing other variables

TEST_VARIABLE = "users"  # Can be "users", "epsilon", "queries" etc.

GROUP_METHODS = {
    "cliques": {"name": "Maximal Cliques", "params": None},
    #"uniform": {"name": "Uniform (size=5)", "params": {"avg_size": 5}},
    #"power_law": {"name": "Power Law (α=2.5)", "params": {"alpha": 2.5, "max_size": 20}},
    #"exponential": {"name": "Exponential (λ=5)", "params": {"scale": 5, "max_size": 20}}
}


def plot_group_size_distribution(groups, filename_prefix):
    """Plot the distribution of group sizes"""
    sizes = [len(group["nodes"]) for group in groups]
    size_counts = defaultdict(int)
    for size in sizes:
        size_counts[size] += 1

    sizes_sorted = sorted(size_counts.items())
    sizes, counts = zip(*sizes_sorted)

    plt.figure(figsize=(8, 6))
    plt.bar(sizes, counts, color='k', alpha=0.7)
    plt.xlabel('Group Size', size=24)
    plt.ylabel('Count', size=24)
    plt.xticks(size=24)
    plt.yticks(size=24)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}.pdf", bbox_inches='tight')
    plt.close()

    # Save data
    with open(f"{filename_prefix}.txt", 'w') as f:
        f.write("Size\tCount\n")
        for size, count in sizes_sorted:
            f.write(f"{size}\t{count}\n")


def generate_groups_uniform(G, avg_size=5, num_groups=None):
    """Generate groups with uniform size distribution"""
    nodes = list(G.nodes())
    random.shuffle(nodes)

    if num_groups is None:
        num_groups = len(nodes) // avg_size

    groups = []
    for i in range(num_groups):
        size = avg_size
        start = i * size
        end = (i + 1) * size
        if start >= len(nodes):
            break
        groups.append(nodes[start:end])

    # Add remaining nodes as individual groups
    remaining = num_groups * avg_size
    if remaining < len(nodes):
        for node in nodes[remaining:]:
            groups.append([node])

    return groups


def generate_groups_power_law(G, alpha=2.5, max_size=20, num_groups=None):
    """Generate groups with power law size distribution"""
    nodes = list(G.nodes())
    random.shuffle(nodes)

    if num_groups is None:
        num_groups = len(nodes) // 5  # Default to roughly 20% of nodes

    groups = []
    node_ptr = 0

    for _ in range(num_groups):
        # Sample group size from power law distribution
        size = min(int(np.random.power(alpha) * max_size) + 1, len(nodes) - node_ptr)
        if size <= 0:
            break
        groups.append(nodes[node_ptr:node_ptr + size])
        node_ptr += size

    # Add remaining nodes as individual groups
    if node_ptr < len(nodes):
        for node in nodes[node_ptr:]:
            groups.append([node])

    return groups


def generate_groups_exponential(G, scale=5, max_size=20, num_groups=None):
    """Generate groups with exponential size distribution"""
    nodes = list(G.nodes())
    random.shuffle(nodes)

    if num_groups is None:
        num_groups = len(nodes) // 5  # Default to roughly 20% of nodes

    groups = []
    node_ptr = 0

    for _ in range(num_groups):
        # Sample group size from exponential distribution
        size = min(int(np.random.exponential(scale)) + 1, len(nodes) - node_ptr)
        if size <= 0:
            break
        groups.append(nodes[node_ptr:node_ptr + size])
        node_ptr += size

    # Add remaining nodes as individual groups
    if node_ptr < len(nodes):
        for node in nodes[node_ptr:]:
            groups.append([node])

    return groups


def convertToXY(p_gis, lat_mean):
    lat = p_gis[0]
    longi = p_gis[1]
    r = 6371000
    x = r * longi * cos(lat_mean)
    y = r * lat
    return x, y


def adjustXY(p, pmin):
    return p[0] - pmin[0], p[1] - pmin[1]


def get_reference_xy(df):
    longs = df['longt'].apply(radians)
    lats = df['lat'].apply(radians)
    # get the lower left and upper right points
    pmax_gis = (max(lats), max(longs))
    pmin_gis = (min(lats), min(longs))
    # get the mean point for GIS to X,Y coordinates mapping
    lat_mean = (pmax_gis[0] + pmin_gis[0]) / 2
    pmax = convertToXY(pmax_gis, lat_mean)
    pmin = convertToXY(pmin_gis, lat_mean)
    if True:
        print(pmax, pmin)
    pmax_adj = adjustXY(pmax, pmin)
    pmin_adj = adjustXY(pmin, pmin)
    if True:
        print(pmax_adj, pmin_adj)
    return lat_mean, pmin


def load_and_filter_data(filepath, lat_range, long_range, max_records_per_user):
    """Load and filter the check-ins data."""
    df = pd.read_csv(filepath, sep='\t', header=None)
    df.columns = ['id', 'time', 'lat', 'longt', 'locId']

    # Filter by geographic coordinates
    df = df[
        (df['lat'] > lat_range[0]) &
        (df['lat'] < lat_range[1]) &
        (df['longt'] > long_range[0]) &
        (df['longt'] < long_range[1])
        ]

    # Filter users with too many records
    return df[df.groupby('id')['id'].transform('size') <= max_records_per_user]


def process_user_locations(df, num_users, lat_mean, pmin):
    """Process user locations and convert to XY coordinates."""
    # Limit to top N users
    uids = df['id'].unique()[:num_users]
    df = df[df['id'].isin(uids)]

    # Aggregate locations
    aggregated_df = df.groupby('id').agg({'lat': 'mean', 'longt': 'mean'}).reset_index()

    # Convert to XY coordinates
    aggregated_df['x'] = aggregated_df.apply(
        lambda row: convertToXY((radians(row['lat']), radians(row['longt'])), lat_mean)[0], axis=1
    )
    aggregated_df['y'] = aggregated_df.apply(
        lambda row: convertToXY((radians(row['lat']), radians(row['longt'])), lat_mean)[1], axis=1
    )

    # Adjust coordinates
    aggregated_df['x'] -= pmin[0]
    aggregated_df['y'] -= pmin[1]

    return aggregated_df, uids


def build_graph(aggregated_df, uids, edges_filepath):
    """Build the network graph from user data and edges."""
    G = nx.Graph()

    # Add nodes with attributes
    for uid in uids:
        user_data = aggregated_df[aggregated_df['id'] == uid]
        if not user_data.empty:
            user_data = user_data.iloc[0]
            G.add_node(uid, x=user_data['x'], y=user_data['y'])
        else:
            G.add_node(uid)

    # Add edges
    edges_df = pd.read_csv(edges_filepath, sep='\t', header=None, names=['user1', 'user2'])
    filtered_edges = edges_df[(edges_df['user1'].isin(uids)) & (edges_df['user2'].isin(uids))]

    G.add_edges_from(filtered_edges.to_records(index=False))

    return G


def process_groups(G, distribution, group_method="cliques", group_params=None):

    """Process cliques and assign methods with attributes."""

    if group_method == "cliques":
        groups = list(nx.find_cliques(G))
    elif group_method == "uniform":
        avg_size = group_params.get("avg_size", 5) if group_params else 5
        groups = generate_groups_uniform(G, avg_size)
    elif group_method == "power_law":
        alpha = group_params.get("alpha", 2.5) if group_params else 2.5
        max_size = group_params.get("max_size", 20) if group_params else 20
        groups = generate_groups_power_law(G, alpha, max_size)
    elif group_method == "exponential":
        scale = group_params.get("scale", 5) if group_params else 5
        max_size = group_params.get("max_size", 20) if group_params else 20
        groups = generate_groups_exponential(G, scale, max_size)
    else:
        raise ValueError(f"Unknown group method: {group_method}")

    # Define the methods
    methods = ["SMC1", "SMC2", "SDP1", "SDP2", "CDP"]

    # Assign methods to groups
    assigned_methods = assign_methods(groups, distribution)

    # Create groups with attributes
    groups_with_attributes = []
    for i, group in enumerate(groups):
        method = assigned_methods[i]
        group_size = len(group)
        error, delay = calculate_attributes(method, group_size)
        groups_with_attributes.append({
            "nodes": group,
            "method": method,
            "error": error,
            "delay": delay
        })

    # Add single-node groups with LDP
    for node in G.nodes():
        groups_with_attributes.append({
            "nodes": [node],
            "method": "LDP",
            "error": 1,
            "delay": 1
        })

    return groups_with_attributes


def assign_methods(groups, distribution):
    """Assign methods to groups based on distribution."""
    num_groups = len(groups)
    method_counts = {method: round(ratio * num_groups) for method, ratio in distribution.items()}

    # Adjust counts to ensure total matches number of groups
    while sum(method_counts.values()) != num_groups:
        for method in method_counts:
            if sum(method_counts.values()) < num_groups:
                method_counts[method] += 1
            else:
                method_counts[method] -= 1

    # Assign methods to groups
    assigned_methods = []
    for method, count in method_counts.items():
        assigned_methods.extend([method] * count)

    random.shuffle(assigned_methods)
    return assigned_methods


def calculate_attributes(method, group_size):
    """Calculate error and delay based on method and group size."""
    if method == "SMC1":
        return 1, group_size * 5
    elif method == "SMC2":
        return 1, group_size * 6
    elif method == "SDP1":
        return 1, 2 * math.log(group_size) * math.log(group_size)
    elif method == "SDP2":
        return group_size ** (1 / 6), 2 * math.log(group_size)
    elif method == "CDP":
        return 1, 1
    raise ValueError("Unknown method")


# Utility functions
def is_within_range(x, y, center, radius):
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2


def add_laplace_noise(value, scale):
    noisy_value = value + np.random.laplace(0, scale / EPS)
    return max(noisy_value, 0)


def generate_random_centers(G, num_centers=NUM_CENTERS):
    x_coords = [data['x'] for _, data in G.nodes(data=True)]
    y_coords = [data['y'] for _, data in G.nodes(data=True)]
    return [
        (np.random.uniform(min(x_coords), max(x_coords)),
         np.random.uniform(min(y_coords), max(y_coords)))
        for _ in range(num_centers)
    ]


# Group selection methods
def greedy_group_selection_initial_version(groups, delay_max):
    uncovered = set().union(*[group["nodes"] for group in groups])
    selected_groups = []
    max_delay = 0

    while uncovered:
        best_group = None
        best_ratio = float('inf')
        best_covered = set()

        for group in groups:
            if group not in selected_groups:
                new_max_delay = max(max_delay, group["delay"])
                if new_max_delay > delay_max:
                    continue

                covered = set(group["nodes"]).intersection(uncovered)
                if not covered:
                    continue

                ratio = group["error"] / len(covered)
                if ratio < best_ratio:
                    best_group = group
                    best_ratio = ratio
                    best_covered = covered

        if not best_group:
            print(f"Warning: Cannot cover all nodes without violating delay_max ({delay_max}).")
            break

        selected_groups.append(best_group)
        uncovered -= best_covered
        max_delay = max(max_delay, best_group["delay"])

    return selected_groups


def greedy_group_selection(groups, delay_max):
    """Optimized greedy group selection using a priority queue (min-heap)."""
    uncovered = set().union(*[group["nodes"] for group in groups])
    selected_groups = []
    max_delay = 0

    # Preprocess: Create a mapping from nodes to their individual groups
    individual_groups = {}
    for group in groups:
        if len(group["nodes"]) == 1:
            node = next(iter(group["nodes"]))
            individual_groups[node] = group

    # Initialize priority queue
    heap = []
    for idx, group in enumerate(groups):
        covered = set(group["nodes"]).intersection(uncovered)
        if covered:
            ratio = group["error"] / len(covered)
            heappush(heap, (ratio, idx, group))

    while uncovered:
        if not heap:
            # Fall back to individual groups
            for node in uncovered:
                individual_group = individual_groups.get(node)
                if individual_group:
                    new_max_delay = max(max_delay, individual_group["delay"])
                    if new_max_delay <= delay_max:
                        selected_groups.append(individual_group)
                        uncovered.remove(node)
                        max_delay = new_max_delay
            break

        # Get best group
        best_ratio, best_idx, best_group = heappop(heap)
        new_max_delay = max(max_delay, best_group["delay"])

        if new_max_delay > delay_max:
            continue

        selected_groups.append(best_group)
        covered = set(best_group["nodes"]).intersection(uncovered)
        uncovered -= covered
        max_delay = new_max_delay

        # Update heap
        new_heap = []
        for ratio, idx, group in heap:
            covered = set(group["nodes"]).intersection(uncovered)
            if covered:
                new_ratio = group["error"] / len(covered)
                heappush(new_heap, (new_ratio, idx, group))
        heap = new_heap

    return selected_groups


def ldp_baseline_group_selection(groups, delay_max):
    return [group for group in groups if group['method'] == 'LDP']


def random_group_selection(groups, delay_max):
    filtered_groups = [group for group in groups if group["delay"] <= delay_max]
    random.shuffle(filtered_groups)

    selected_groups = []
    covered_nodes = set()

    for group in filtered_groups:
        new_nodes = set(group["nodes"]) - covered_nodes
        if new_nodes:
            selected_groups.append(group)
            covered_nodes.update(new_nodes)
        if len(covered_nodes) == G.number_of_nodes():
            break

    return selected_groups


def largest_group_first(groups, delay_max):
    filtered_groups = sorted(
        [group for group in groups if group["delay"] <= delay_max],
        key=lambda x: len(x["nodes"]), reverse=True
    )

    selected_groups = []
    covered_nodes = set()

    for group in filtered_groups:
        new_nodes = set(group["nodes"]) - covered_nodes
        if new_nodes:
            selected_groups.append(group)
            covered_nodes.update(new_nodes)
        if len(covered_nodes) == G.number_of_nodes():
            break

    return selected_groups


def smallest_error_first(groups, delay_max):
    filtered_groups = sorted(
        [group for group in groups if group["delay"] <= delay_max],
        key=lambda x: x["error"]
    )

    selected_groups = []
    covered_nodes = set()

    for group in filtered_groups:
        new_nodes = set(group["nodes"]) - covered_nodes
        if new_nodes:
            selected_groups.append(group)
            covered_nodes.update(new_nodes)
        if len(covered_nodes) == G.number_of_nodes():
            break

    return selected_groups


def balanced_error_coverage(groups, delay_max):
    filtered_groups = [group for group in groups if group["delay"] <= delay_max]
    selected_groups = []
    covered_nodes = set()

    while len(covered_nodes) < G.number_of_nodes():
        best_group = None
        best_ratio = 0

        for group in filtered_groups:
            if group not in selected_groups:
                new_nodes = set(group["nodes"]) - covered_nodes
                if new_nodes:
                    ratio = len(new_nodes) / group["error"]
                    if ratio > best_ratio:
                        best_group = group
                        best_ratio = ratio

        if best_group:
            selected_groups.append(best_group)
            covered_nodes.update(best_group["nodes"])
        else:
            break

    return selected_groups


# Evaluation functions
def calculate_center_results(selected_groups, G, center, radius, num_rounds=NUM_ROUNDS):
    correct_count = sum(
        sum(1 for node in group["nodes"]
            if is_within_range(G.nodes[node]['x'], G.nodes[node]['y'], center, radius))
        for group in selected_groups
    )

    noisy_results = []
    for _ in range(num_rounds):
        noisy_total = sum(
            add_laplace_noise(
                sum(1 for node in group["nodes"]
                    if is_within_range(G.nodes[node]['x'], G.nodes[node]['y'], center, radius)),
                group["error"]
            )
            for group in selected_groups
        )
        noisy_results.append(noisy_total)

    differences = [abs(correct_count - noisy) for noisy in noisy_results]
    return correct_count, np.mean(differences)


def evaluate_results(G, groups_with_attributes, centers, radius, num_rounds=NUM_ROUNDS):
    results = {}
    selection_methods = {
        "Greedy": greedy_group_selection,
        "LDP": ldp_baseline_group_selection,
        "Random": random_group_selection,
        "Largest Group": largest_group_first,
        "Smallest Error": smallest_error_first,
    }

    for delay_max in DELAY_MAX_VALUES:
        method_results = {}
        for method_name, method_func in selection_methods.items():
            start_time = time.time()
            selected_groups = method_func(groups_with_attributes, delay_max)
            runtime = time.time() - start_time

            center_results = [
                calculate_center_results(selected_groups, G, center, radius, num_rounds)
                for center in centers
            ]

            method_results[method_name] = {
                "results": center_results,
                "runtime": runtime,
                "num_groups": len(selected_groups)
            }

        results[delay_max] = method_results

    return results


def save_plot_data(results, filename_prefix):
    """Saves the numerical data used in plots to a text file"""
    txt_filename = f"{filename_prefix}.txt"
    with open(txt_filename, 'w') as f:
        f.write("Delay Max\tGreedy Error\tLDP Error\tRandom Error\tLargest Group Error\n")
        for delay_max in sorted(results.keys()):
            errors = {}
            for method in results[delay_max]:
                avg_error = sum(r[1] for r in results[delay_max][method]["results"]) / len(
                    results[delay_max][method]["results"])
                errors[method] = avg_error

            f.write(f"{delay_max}\t"
                    f"{errors.get('Greedy', 'NA'):.2f}\t"
                    f"{errors.get('LDP', 'NA'):.2f}\t"
                    f"{errors.get('Random', 'NA'):.2f}\t"
                    f"{errors.get('Largest Group', 'NA'):.2f}\n")


def plot_results(results, eps, centers, title_suffix=""):
    # Setup figure with spines style
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Prepare data
    delay_max_values = sorted(results.keys())
    N = len(delay_max_values)
    x = np.arange(N)

    # Extract errors for each method
    errors = {
        'Greedy': [],
        'LDP': [],
        'Random': [],
        'Largest Group': []
    }

    for delay_max in delay_max_values:
        for method in errors.keys():
            if method in results[delay_max]:
                avg_error = sum(r[1] for r in results[delay_max][method]["results"]) / len(centers)
                errors[method].append(avg_error)
            else:
                errors[method].append(np.nan)

    # Plot with reference figure's style
    p1 = plt.plot(x, errors['Greedy'], color='k', marker='o', markersize=12, linestyle='-')
    p2 = plt.plot(x, errors['LDP'], color='k', marker='v', linestyle='--',
                  markerfacecolor='none', markersize=12)
    p3 = plt.plot(x, errors['Random'], color='k', marker='s', linestyle=':',
                  markerfacecolor='none', markersize=12)
    p4 = plt.plot(x, errors['Largest Group'], color='k', marker='^', linestyle='-.',
                  markerfacecolor='none', markersize=12)

    # Axis formatting
    plt.xlabel('Delay Max', size=24)
    plt.ylabel('Average Error', size=24)
    plt.xticks(x, [str(d) for d in delay_max_values], size=24)

    # Auto-adjust y-ticks based on data range
    y_min = min([min(e) for e in errors.values() if len(e) > 0])
    y_max = max([max(e) for e in errors.values() if len(e) > 0])
    plt.yticks(np.linspace(y_min, y_max, 5), size=24)

    # Legend
    plt.legend(
        (p1[0], p2[0], p3[0], p4[0]),
        ('Greedy', 'LDP', 'Random', 'Largest Group'),
        frameon=False,
        fontsize=24,
        ncol=2,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.1)
    )


    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    # Save with matching filename
    filename = f"Effectiveness{title_suffix}.pdf"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    # Save data
    save_plot_data(results, filename.replace('.pdf', ''))

def plot_runtime(results, eps, centers, title_suffix=""):
    # Setup figure with spines style
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Prepare data
    delay_max_values = sorted(results.keys())
    N = len(delay_max_values)
    x = np.arange(N)

    # Extract runtimes for each method
    runtimes = {
        'Greedy': [],
        'LDP': [],
        'Random': [],
        'Largest Group': []
    }

    for delay_max in delay_max_values:
        for method in runtimes.keys():
            if method in results[delay_max]:
                runtime = results[delay_max][method]["runtime"]
                runtimes[method].append(runtime)
            else:
                runtimes[method].append(np.nan)

    # Plot with reference figure's style
    p1 = plt.plot(x, runtimes['Greedy'], color='k', marker='o', markersize=12, linestyle='-')
    p2 = plt.plot(x, runtimes['LDP'], color='k', marker='v', linestyle='--',
                  markerfacecolor='none', markersize=12)
    p3 = plt.plot(x, runtimes['Random'], color='k', marker='s', linestyle=':',
                  markerfacecolor='none', markersize=12)
    p4 = plt.plot(x, runtimes['Largest Group'], color='k', marker='^', linestyle='-.',
                  markerfacecolor='none', markersize=12)

    # Axis formatting
    plt.xlabel('Delay Max', size=24)
    plt.ylabel('Runtime (seconds)', size=24)
    plt.xticks(x, [str(d) for d in delay_max_values], size=24)

    # Auto-adjust y-ticks based on data range
    y_min = min([min(t) for t in runtimes.values() if len(t) > 0])
    y_max = max([max(t) for t in runtimes.values() if len(t) > 0])
    plt.yticks(np.linspace(y_min, y_max, 5), size=24)

    # Legend
    plt.legend(
        (p1[0], p2[0], p3[0], p4[0]),
        ('Greedy', 'LDP', 'Random', 'Largest Group'),
        frameon=False,
        fontsize=24,
        ncol=2,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.3)
    )

    # Adjust layout to make room for legend
    plt.tight_layout()

    # Save with matching filename
    filename = f"Efficiency{title_suffix}.pdf"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    # Save data
    save_runtime_data(results, filename.replace('.pdf', ''))


def save_runtime_data(results, filename_prefix):
    """Saves the runtime data used in plots to a text file"""
    txt_filename = f"{filename_prefix}.txt"
    with open(txt_filename, 'w') as f:
        f.write("Delay Max\tGreedy Runtime\tLDP Runtime\tRandom Runtime\tLargest Group Runtime\n")
        for delay_max in sorted(results.keys()):
            times = {}
            for method in results[delay_max]:
                times[method] = results[delay_max][method]["runtime"]

            f.write(f"{delay_max}\t"
                    f"{times.get('Greedy', 'NA'):.4f}\t"
                    f"{times.get('LDP', 'NA'):.4f}\t"
                    f"{times.get('Random', 'NA'):.4f}\t"
                    f"{times.get('Largest Group', 'NA'):.4f}\n")


def evaluate_all_epsilons(G, groups_with_attributes, group_method=""):
    centers = generate_random_centers(G)

    for eps in EPS_VALUES:
        print(f"\nEvaluating with epsilon = {eps}")
        global EPS  # Temporarily modify the global EPS value
        EPS = eps

        results = evaluate_results(G, groups_with_attributes, centers, RADIUS)
        current_date = datetime.now().strftime("%b%d")

        # Plot both error and runtime
        plot_results(results, eps, centers, f"_{current_date}_eps_{eps:.1f}_{group_method}")
        plot_runtime(results, eps, centers, f"_{current_date}_eps_{eps:.1f}_{group_method}")

        print("\nEvaluation Summary:")
        for delay_max, method_data in results.items():
            print(f"\nDelay Max: {delay_max}")
            for method_name, data in method_data.items():
                avg_error = sum(center_data[1] for center_data in data["results"]) / len(centers)
                print(
                    f"{method_name}: Avg Error={avg_error:.2f}, Runtime={data['runtime']:.4f}s, Groups={data['num_groups']}")



# Modify the main execution
if __name__ == '__main__':
    # Load and process data
    df = load_and_filter_data(CHECKINS_FILE, LAT_RANGE, LONG_RANGE, MAX_RECORDS_PER_USER)
    lat_mean, pmin = get_reference_xy(df)

    if TEST_VARIABLE == "users":
        global EPS
        EPS = DEFAULT_EPS
        for num_users in USER_COUNTS:
            print(f"\nTesting with {num_users} users")
            aggregated_df, uids = process_user_locations(df, num_users, lat_mean, pmin)
            G = build_graph(aggregated_df, uids, EDGES_FILE)

            for group_method, method_info in GROUP_METHODS.items():
                print(f"\nUsing group method: {method_info['name']}")
                groups_with_attributes = process_groups(
                    G, DISTRIBUTIONS["uniform"],
                    group_method, method_info["params"]
                )

                centers = generate_random_centers(G)
                results = evaluate_results(G, groups_with_attributes, centers, RADIUS)

                current_date = datetime.now().strftime("%b%d")
                suffix = f"_{current_date}_users_{num_users}_{group_method}"
                plot_results(results, DEFAULT_EPS, centers, suffix)
                plot_runtime(results, DEFAULT_EPS, centers, suffix)

                # Plot group size distribution
                plot_group_size_distribution(groups_with_attributes, f"GroupSizeDistribution{suffix}")

    elif TEST_VARIABLE == "epsilon":
        aggregated_df, uids = process_user_locations(df, DEFAULT_USERS, lat_mean, pmin)
        G = build_graph(aggregated_df, uids, EDGES_FILE)

        for group_method, method_info in GROUP_METHODS.items():
            print(f"\nUsing group method: {method_info['name']}")
            groups_with_attributes = process_groups(
                G, DISTRIBUTIONS["uniform"],
                group_method, method_info["params"]
            )

            evaluate_all_epsilons(G, groups_with_attributes, group_method)

            # Plot group size distribution
            suffix = f"_{datetime.now().strftime('%b%d')}_eps_{DEFAULT_EPS:.1f}_{group_method}"
            plot_group_size_distribution(groups_with_attributes, f"GroupSizeDistribution{suffix}")

