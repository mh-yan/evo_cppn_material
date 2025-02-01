"""Implements the core evolution algorithm."""

from neat.math_util import mean
from neat.reporting import ReporterSet
from neat.spea2 import calculate_fitness, get_archive, get_nondominated_and_dominating
import matplotlib.pyplot as plt
import numpy as np

import pickle
import os


class CompleteExtinctionException(Exception):
    pass


class Population(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, config, initial_state=None):
        self.reporters = ReporterSet()
        self.config = config
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(
            config.reproduction_config, self.reporters, stagnation
        )
        if config.fitness_criterion == "max":
            self.fitness_criterion = max
        elif config.fitness_criterion == "min":
            self.fitness_criterion = min
        elif config.fitness_criterion == "mean":
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion)
            )

        if initial_state is None:
            # Create a population from scratch, then partition into speconfig.pop_sizecies.
            self.population = self.reproduction.create_new(
                config.genome_type, config.genome_config, config.pop_size
            )
            self.species = config.species_set_type(
                config.species_set_config, self.reporters
            )
            self.generation = 0
            self.species.speciate(self.config, self.population, self.generation)

        else:
            self.population, self.species, self.generation = initial_state

        self.best_genome = None

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)

    def termination_threshold(self):
        if not self.config.no_fitness_termination:
            # End if the fitness threshold is reached.
            fv = self.fitness_criterion(g.fitness for g in self.population.values())
            if fv >= self.config.fitness_threshold:
                self.reporters.found_solution(self.config, self.generation, best)
                return True
        return False

    def neat_reproduce_speciate(self):
        self.population = self.reproduction.reproduce(
            self.config,
            self.species,
            self.config.pop_size,
            self.generation,
        )
        self.species.speciate(self.config, self.population, self.generation)

    def reference_points(self, M, p):

        def generator(r_points, M, Q, T, D):
            points = []
            if D == M - 1:
                r_points[D] = Q / (1.0 * T)
                points.append(r_points)
            elif D != M - 1:
                for i in range(Q + 1):
                    r_points[D] = i / T
                    points.extend(generator(r_points.copy(), M, Q - i, T, D + 1))
            return points

        ref_points = np.array(generator(np.zeros(M), M, p, p, 0))
        print(f"generate {len(ref_points)} refs")
        return ref_points

    # Function: Nearest Vectors

    def nearest_vectors(self, weights):
        sorted_cosine = -np.sort(-np.dot(weights, weights.T), axis=1)
        arccosine_weights = np.arccos(np.clip(sorted_cosine[:, 1], 0, 1))
        return arccosine_weights

    def adaptation(self, pop, vectors, vectors_, M):
        fits_all = [g.fitnesses for g in pop.values() if g.is_normal == 1]
        # fits_all = [g.fitnesses for g in pop.values()]
        fits_all = np.array(fits_all)
        z_min = np.min(fits_all, axis=0)
        z_max = np.max(fits_all, axis=0)
        vectors = vectors_ * (z_max - z_min)
        vectors = vectors / (np.linalg.norm(vectors, axis=1).reshape(-1, 1))
        neighbours = self.nearest_vectors(vectors)
        return vectors, neighbours

    def save_and_plot_fitnesses(self, fitnesses, vectors):
        # save  file
        save_dir = self.sava_k_gen(
            k=self.generation,
            task_name=self.task_name,
            out_path=self.out_path,
            cur_times=self.cur_times,
        )
        z_min = np.min(fitnesses, axis=0)
        fitnesses_translate = fitnesses - z_min
        colors = np.random.normal(0, 1, len(fitnesses))

        # Plot translated fitnesses
        plt.figure()
        plt.scatter(
            fitnesses_translate[:, 0],
            fitnesses_translate[:, 1],
            c=colors,
            s=10,
            cmap="viridis",
        )

        for vec in vectors:
            plt.quiver(
                0,
                0,
                vec[0],
                vec[1],
                angles="xy",
                scale_units="xy",
                scale=1,
                color="r",
                alpha=0.5,
                width=0.003,
            )
        plt.show()
        plt.savefig("./fs_translate.png")
        plt.close()

        # Plot original fitnesses
        plt.figure()
        plt.scatter(-fitnesses[:, 0], fitnesses[:, 1], c=colors, s=10, cmap="tab10")
        plt.xlabel("E")
        plt.ylabel("nu")
        plt.show()
        plt.savefig(f"{save_dir}/fs.png")
        plt.savefig("./fs.png")
        plt.close()
        print(f"min -Eeff, nueff is {[z_min[0], z_min[1]]}")

    def rvea_reproduce(
        self,
        fitness_function,
        config,
        pop,
        pop_size,
        vectors,
        neighbors,
        alpha,
        t,
        t_max,
    ):
        self.population = self.reproduction.reproduce_rvea(
            fitness_function,
            config,
            pop,
            pop_size,
            vectors,
            neighbors,
            alpha,
            t,
            t_max,
            self.species,
        )

        fitnesses = [g.fitnesses for g in self.population.values() if g.is_normal == 1]
        # fitnesses = []

        # color_map = []
        # num_species = len(self.species.species)
        # colors_s = np.linspace(0, 1, num_species)  # 为每个种群生成颜色
        # s_list = list(self.species.species.values())
        # sid_idx_set = {}
        # for i in range(num_species):
        #     s = s_list[i]
        #     sid_idx_set[s.key] = i
        # colors_map = []
        # for i, g in enumerate(self.population.values()):
        #     if g.is_normal == 1:
        #         fitnesses.append(g.fitnesses)
        #         sid = self.species.get_species_id(g.key)
        #         colors_map.append(colors_s[sid_idx_set[sid]])
        fitnesses = np.array(fitnesses)
        self.save_and_plot_fitnesses(fitnesses, vectors)

    def handle_extinction(self):
        if not self.species.species:
            self.reporters.complete_extinction()

            if self.config.reset_on_extinction:
                self.population = self.reproduction.create_new(
                    self.config.genome_type,
                    self.config.genome_config,
                    self.config.pop_size,
                )
            else:
                raise CompleteExtinctionException()

    def sava_k_gen(self, k, task_name, out_path, cur_times):
        output_dir = f"{out_path}/{task_name}_{cur_times}/output{k}"
        os.makedirs(output_dir, exist_ok=True)
        for i, g in enumerate(list(self.population.values())):
            pickle.dump(g, open(f"{output_dir}/genome{i}.pkl", "wb"))
        return output_dir

    def run(self, fitness_function, n, task_name, out_path, cur_times):
        self.task_name = task_name
        self.out_path = out_path
        self.cur_times = cur_times
        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError(
                "Cannot have no generational limit with no fitness termination"
            )
        k = 0
        self.nega_possion_g = {}
        while n is None or k < n:
            k += 1
            self.generation += 1
            self.reporters.start_generation(self.generation)
            if k == 1:
                count = 0
                fitness_function(self.population, self.config, k)
                self.vectors = self.reference_points(
                    M=2, p=int(self.config.pop_size // 2)
                )
                print(
                    f"max xyz of w is {np.max(self.vectors[:,0])}{np.max(self.vectors[:,1])}"
                )
                self.vectors = self.vectors / np.linalg.norm(self.vectors)
                vectors_ = np.copy(self.vectors)
                self.neighbours = self.nearest_vectors(self.vectors)
            # 单双切换，需要在第一次就调整vector
            if count % 15 == 0 and k < 1000:
                self.vectors, self.neighbours = self.adaptation(
                    self.population, self.vectors, vectors_, 2
                )
                print("change vectors")
            self.rvea_reproduce(
                fitness_function,
                config=self.config,
                pop=self.population,
                pop_size=self.config.pop_size,
                vectors=self.vectors,
                neighbors=self.neighbours,
                t=count,
                t_max=n,
                alpha=1.2,
            )
            count = count + 1
