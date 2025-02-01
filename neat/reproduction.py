"""
Handles creation of genomes, either from scratch or by sexual or
asexual reproduction from parents.
"""

import math
import random
from itertools import count
import copy
from neat.config import ConfigParameter, DefaultClassConfig
from neat.math_util import mean
from neat.spea2 import dominates
import numpy as np
from collections import defaultdict, OrderedDict


class DefaultReproduction(DefaultClassConfig):
    """
    Implements the default NEAT-python reproduction scheme:
    explicit fitness sharing with fixed-time species stagnation.
    """

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(
            param_dict,
            [
                ConfigParameter("elitism", int, 0),
                ConfigParameter("survival_threshold", float, 0.2),
                ConfigParameter("min_species_size", int, 1),
            ],
        )

    def __init__(self, config, reporters, stagnation):
        # pylint: disable=super-init-not-called
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)
        self.stagnation = stagnation
        self.ancestors = {}
        self.abnormal_genome = OrderedDict()

    def create_new(self, genome_type, genome_config, num_genomes):
        new_genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
            self.ancestors[key] = tuple()

        return new_genomes

    @staticmethod
    def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        """Compute the proper number of offspring per species (proportional to fitness)."""
        af_sum = sum(adjusted_fitness)

        spawn_amounts = []
        for af, ps in zip(adjusted_fitness, previous_sizes):
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * pop_size)
            else:
                s = min_species_size

            d = (s - ps) * 0.5
            c = int(round(d))
            spawn = ps
            if abs(c) > 0:
                spawn += c
            elif d > 0:
                spawn += 1
            elif d < 0:
                spawn -= 1

            spawn_amounts.append(spawn)

        # Normalize the spawn amounts so that the next generation is roughly
        # the population size requested by the user.
        total_spawn = sum(spawn_amounts)
        norm = pop_size / total_spawn
        spawn_amounts = [
            max(min_species_size, int(round(n * norm))) for n in spawn_amounts
        ]

        return spawn_amounts

    def select_child(self, pop, M, vectors, neighbors, alpha, t, t_max):
        # 找到{min fi}=zmin
        all_f = [g.fitnesses for g in pop.values()]
        all_f = np.array(all_f)
        z_min = np.min(all_f, axis=0)
        translate_f = all_f - z_min
        cos = np.dot(translate_f, vectors.T) / (
            np.linalg.norm(translate_f, axis=1).reshape(-1, 1) + 1e-21
        )
        arc_c = np.arccos(np.clip(cos, 0, 1))
        idx = np.argmax(cos, axis=1)
        niche = dict(zip(np.arange(vectors.shape[0]), [[]] * vectors.shape[0]))
        idx_u = set(idx)
        for i in idx_u:
            niche.update({i: list(np.where(idx == i)[0])})
        idx_ = []
        niche_len = [len(ni) for ni in niche.values()]
        all_gs = [g for g in pop.values()]
        all_gs = np.array(all_gs)
        for i in range(0, vectors.shape[0]):
            if len(niche[i]) != 0:
                individual = niche[i]
                niche_genome = {indv: all_gs[indv] for indv in individual}
                niche_genome_violate = {
                    indv: g for indv, g in niche_genome.items() if g.violate_num > 0
                }
                if len(niche_genome_violate) == len(niche_genome):
                    min_idx = min(
                        niche_genome_violate,
                        key=lambda indv: niche_genome_violate[indv].violate_num,
                    )
                    idx_.append(min_idx)
                else:
                    arc_c_ind = arc_c[individual, i]
                    arc_c_ind = arc_c_ind / neighbors[i]
                    d = np.linalg.norm(translate_f[individual, :], axis=1) * (
                        1 + M * ((t / t_max) ** alpha) * arc_c_ind
                    )
                    idx_adp = np.argmin(d)
                    idx_.append(individual[idx_adp])

        select_gs = all_gs[idx_]
        niches_g = {i: all_gs[indivs] for i, indivs in niche.items()}
        return {g.key: g for g in select_gs}, niches_g

    def cal_reproduce_rate(self, num_niche, pop_size):
        scale_factor = 1 / 3.0
        num_niche = np.array(num_niche)
        num_niche = [
            (1 / count) ** scale_factor if count > 0 else 0 for count in num_niche
        ]
        sum_num_niche = np.sum(num_niche)
        num_niche = np.ceil((num_niche / sum_num_niche) * pop_size)
        num_niche = [count if count > 0 else 2 for count in num_niche]
        return num_niche

    def merge_niche(self, niches_g, merge_num):
        before_len = len(niches_g)
        # 初始化一个新的字典来存储合并后的niches
        after_len = int(np.ceil(len(niches_g) / merge_num))
        combined_niches_g = {}
        for i in range(after_len):
            merge_values = []
            for j in range(merge_num):
                if i * merge_num + j < before_len:
                    merge_values.extend(niches_g[i * merge_num + j])
                else:
                    break
            combined_niches_g[i] = merge_values
        self.niches_g = combined_niches_g

    def breed_species(
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
        species,
    ):
        # 当前species 数量
        cur_species_len = []
        for sid, s in species.species.items():
            cur_species_len.append(len(s.members))
        print(f"cur species len is {cur_species_len}")
        # 下次species数量
        repro_num = self.cal_reproduce_rate(cur_species_len, pop_size)
        print(f"len reprod {repro_num}")
        # 种群内breed
        for i, (sid, s) in enumerate(species.species.items()):
            cur_smembers = list(s.members.values())
            s.members = {}
            species.species[s.key] = s
            while repro_num[i] != 0:
                parent1 = random.choice(cur_smembers)
                parent2 = random.choice(cur_smembers)
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent1, parent2, config.genome_config)
                child.mutate(config.genome_config)
                pop[gid] = child
                repro_num[i] -= 1

        # select
        fitness_function(pop, config, t)
        pop, niches_g = self.select_child(pop, 2, vectors, neighbors, alpha, t, t_max)
        species.speciate(config, pop, t)
        return pop

    def breed_rc(
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
        speciecs,
    ):
        num_merge_niche = config.merge_num
        print(f"global_rate is {config.global_rate}")
        if t != 0:
            # several niche need to merge
            niches_len = [len(ni) for ni in self.niches_g.values()]
            repro_num_niche = np.ceil(self.cal_reproduce_rate(niches_len, pop_size))
            print(f"len before {niches_len}")
            print(f"len reprod {repro_num_niche}")
            # local_reproduce
            for i in range(len(niches_len)):
                repro_num = int(repro_num_niche[i])
                while repro_num > 0:
                    if len(self.niches_g[i]) != 0:
                        parent1 = random.choice(list(self.niches_g[i]))
                        parent2 = random.choice(list(self.niches_g[i]))
                        gid = next(self.genome_indexer)
                        child = config.genome_type(gid)
                        child.configure_crossover(
                            parent1, parent2, config.genome_config
                        )
                        child.mutate(config.genome_config)
                    else:
                        gid = next(self.genome_indexer)
                        child = config.genome_type(gid)
                        child.configure_new(config.genome_config)
                        child.mutate(config.genome_config)
                    pop[gid] = child
                    repro_num -= 1
        # select
        fitness_function(pop, config, t)
        pop, niches_g = self.select_child(pop, 2, vectors, neighbors, alpha, t, t_max)
        self.merge_niche(niches_g, num_merge_niche)
        return pop

    def breed_rc_mix(
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
        speciecs,
    ):
        num_merge_niche = config.merge_num
        print(f"global_rate is {config.global_rate}")
        if t != 0:
            # several niche need to merge
            niches_len = [len(ni) for ni in self.niches_g.values()]
            local_rate = 1 - config.global_rate
            repro_num_niche = np.ceil(
                self.cal_reproduce_rate(niches_len, int(pop_size * local_rate))
            )
            print(f"len before {niches_len}")
            print(f"len reprod {repro_num_niche}")
            # niche_offspring = defaultdict(list)
            print("global num is ", int(pop_size * config.global_rate))
            print(f"len old{len(pop)}")
            old_p = copy.deepcopy(pop)
            old_p.update(self.abnormal_genome)
            print(f"after len old{len(old_p)}")
            # global_reproduce from old and abnormal
            for i in range(int(pop_size * config.global_rate)):
                parent1 = random.choice(list(old_p.values()))
                parent2 = random.choice(list(old_p.values()))
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent1, parent2, config.genome_config)
                child.mutate(config.genome_config)
                pop[gid] = child
            # local_reproduce
            for i in range(len(niches_len)):
                repro_num = int(repro_num_niche[i])
                while repro_num > 0:
                    if len(self.niches_g[i]) != 0:
                        parent1 = random.choice(list(self.niches_g[i]))
                        parent2 = random.choice(list(self.niches_g[i]))
                        gid = next(self.genome_indexer)
                        child = config.genome_type(gid)
                        child.configure_crossover(
                            parent1, parent2, config.genome_config
                        )
                        child.mutate(config.genome_config)
                    else:
                        gid = next(self.genome_indexer)
                        child = config.genome_type(gid)
                        child.configure_new(config.genome_config)
                        child.mutate(config.genome_config)
                    pop[gid] = child
                    repro_num -= 1

        # select
        fitness_function(pop, config, t)
        n_p = {}
        ab_p = {}
        # seperate
        for k, g in pop.items():
            if g.is_normal == 1:
                n_p[k] = g
            else:
                self.abnormal_genome[k] = g
                if len(self.abnormal_genome) > pop_size / 2:
                    self.abnormal_genome.popitem(last=False)
        pop, niches_g = self.select_child(n_p, 2, vectors, neighbors, alpha, t, t_max)
        self.merge_niche(niches_g, num_merge_niche)
        return pop

    def breed_random(
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
        speciecs,
    ):
        old_p = copy.deepcopy(pop)
        while len(pop) < 2 * pop_size:
            parent1 = random.choice(list(old_p.values()))
            parent2 = random.choice(list(old_p.values()))
            gid = next(self.genome_indexer)
            child = config.genome_type(gid)
            child.configure_crossover(parent1, parent2, config.genome_config)
            child.mutate(config.genome_config)
            pop[gid] = child
        fitness_function(pop, config, t)
        pop, niches_g = self.select_child(pop, 2, vectors, neighbors, alpha, t, t_max)
        return pop

    def reproduce_rvea(
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
        speciecs,
    ):
        pop = self.breed_rc_mix(
            fitness_function,
            config,
            pop,
            pop_size,
            vectors,
            neighbors,
            alpha,
            t,
            t_max,
            speciecs,
        )
        return pop

    def reproduce(self, config, species, pop_size, generation):

        all_fitnesses = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                all_fitnesses.extend(m.fitness for m in stag_s.members.values())
                remaining_species.append(stag_s)

        # No species left.
        if not remaining_species:
            species.species = {}
            return {}  # was []

        # Find minimum/maximum fitness across the entire population, for use in
        # species adjusted fitness computation.
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        # Do not allow the fitness range to be zero, as we divide by it below.
        # TODO: The ``1.0`` below is rather arbitrary, and should be configurable.
        fitness_range = max(1.0, max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = mean([m.fitness for m in afs.members.values()])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses)  # type: float
        self.reporters.info(
            "Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness)
        )

        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.reproduction_config.min_species_size
        # Isn't the effective min_species_size going to be max(min_species_size,
        # self.reproduction_config.elitism)? That would probably produce more accurate tracking
        # of population sizes and relative fitnesses... doing. TODO: document.
        min_species_size = max(min_species_size, self.reproduction_config.elitism)
        spawn_amounts = self.compute_spawn(
            adjusted_fitnesses, previous_sizes, pop_size, min_species_size
        )

        new_population = {}
        species.species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max(spawn, self.reproduction_config.elitism)

            assert spawn > 0

            # The species has at least one member for the next generation, so retain it.
            old_members = list(s.members.items())
            s.members = {}
            species.species[s.key] = s

            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            # Transfer elites to new generation.
            if self.reproduction_config.elitism > 0:
                for i, m in old_members[: self.reproduction_config.elitism]:
                    new_population[i] = m
                    spawn -= 1

            if spawn <= 0:
                continue

            # Only use the survival threshold fraction to use as parents for the next generation.
            repro_cutoff = int(
                math.ceil(
                    self.reproduction_config.survival_threshold * len(old_members)
                )
            )
            # Use at least two parents no matter what the threshold fraction result is.
            repro_cutoff = max(repro_cutoff, 2)
            old_members = old_members[:repro_cutoff]

            # Randomly choose parents and produce the number of offspring allotted to the species.
            while spawn > 0:
                spawn -= 1

                parent1_id, parent1 = random.choice(old_members)
                parent2_id, parent2 = random.choice(old_members)

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent1, parent2, config.genome_config)
                child.mutate(config.genome_config)
                # TODO: if config.genome_config.feed_forward, no cycles should exist
                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)

        return new_population
