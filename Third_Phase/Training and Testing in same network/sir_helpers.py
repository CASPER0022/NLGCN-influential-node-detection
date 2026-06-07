import random

# Global adjacency dictionary shared across runs in a worker process
global_adj = {}

def init_worker_from_file(dataset_path):
    """
    Called once when each worker process starts.
    Loads the dataset using pure Python (0% external library dependencies).
    This ensures it runs perfectly even if the background process launches
    using a global Python interpreter that lacks numpy/scipy!
    """
    global global_adj
    adj = {}
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip comments or empty lines
                if line.startswith(('%', '#', '//')) or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        u = int(parts[0])
                        v = int(parts[1])
                        adj.setdefault(u, []).append(v)
                        adj.setdefault(v, []).append(u)
                    except ValueError:
                        continue
        global_adj = adj
    except Exception as e:
        global_adj = {}

def SIR_simulation_opt(seed, beta, mu, steps=1000):
    global global_adj
    infected = {seed}
    infected_so_far = {seed}
    recovered_count = 0

    for _ in range(steps):
        new_infected = set()
        new_recovered = set()
        
        for node in infected:
            for nbr in global_adj.get(node, []):
                if nbr not in infected_so_far:
                    if random.random() < beta:
                        new_infected.add(nbr)
            
            if random.random() < mu:
                new_recovered.add(node)

        infected_so_far.update(new_infected)
        infected |= new_infected
        infected -= new_recovered
        recovered_count += len(new_recovered)

        if len(infected) == 0:
            break

    return recovered_count

def simulate_node(args):
    # args is just (node, beta, mu, runs) -- 100% picklable and standard Python types
    node, beta, mu, runs = args
    spread = 0
    for _ in range(runs):
        spread += SIR_simulation_opt(node, beta, mu)
    return spread / runs
