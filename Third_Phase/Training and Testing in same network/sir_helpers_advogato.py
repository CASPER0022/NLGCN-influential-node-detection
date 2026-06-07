import numpy as np
import random

def SIR_simulation_opt(G_edges, seed, beta, mu, steps=1000):
    # Reconstruct local adjacency list for extremely fast thread-safe lookups
    adj = {}
    for u, v in G_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
        
    infected = {seed}
    infected_so_far = {seed}
    recovered_count = 0

    for _ in range(steps):
        new_infected = set()
        new_recovered = set()
        
        for node in infected:
            for nbr in adj.get(node, []):
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
    G_edges, node, beta, mu, runs = args
    spread = 0
    for _ in range(runs):
        spread += SIR_simulation_opt(G_edges, node, beta, mu)
    return spread / runs
