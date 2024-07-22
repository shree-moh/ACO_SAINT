import numpy as np

class AntColonyOptimizer:
    def __init__(self, objective_function, search_space, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        self.objective_function = objective_function
        self.search_space = search_space
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha  # Influence of pheromone
        self.beta = beta  # Influence of heuristic information

        # Initialize pheromones for each hyperparameter in the search space
        self.pheromones = {key: np.ones(len(values)) for key, values in search_space.items()}

    def _select_param(self, pheromones, values):
        # Calculate probabilities for each value based on pheromone levels and heuristic information
        pheromone_levels = np.array(pheromones)
        probabilities = pheromone_levels ** self.alpha
        probabilities /= probabilities.sum()
        return np.random.choice(values, p=probabilities)

    def _generate_ant_params(self):
        ant_params = {}
        for param, values in self.search_space.items():
            selected_value = self._select_param(self.pheromones[param], values)
            ant_params[param] = selected_value
        return ant_params

    def optimize(self):
        best_params = None
        best_score = float('-inf')

        for iteration in range(self.n_iterations):
            all_params = []
            all_scores = []

            for _ in range(self.n_ants):
                params = self._generate_ant_params()
                score = self.objective_function(params)
                all_params.append(params)
                all_scores.append(score)

                if score > best_score:
                    best_score = score
                    best_params = params

            # Update pheromones
            sorted_indices = np.argsort(all_scores)[-self.n_best:]
            for param in self.search_space:
                self.pheromones[param] *= self.decay
                for i in sorted_indices:
                    value_index = self.search_space[param].index(all_params[i][param])
                    self.pheromones[param][value_index] += 1.0

        return best_params, best_score
