"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""

from multiprocessing import Pool


class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None, maxtasksperchild=None):
        """
        eval_function should take one argument, a tuple of (genome object, config object),
        and return a single float (the genome's fitness).
        """
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(processes=num_workers, maxtasksperchild=maxtasksperchild)

    def __del__(self):
        self.pool.close()
        self.pool.join()
        self.pool.terminate()

    def evaluate(self, genomes, config, gen):
        jobs = []
        for genome in list(genomes.values()):
            jobs.append(
                self.pool.apply_async(self.eval_function, (genome, config, gen))
            )

        # assign the fitness back to each genome
        for job, genome in zip(jobs, genomes.values()):
            fits, outputs, is_normal, violate_num = job.get(timeout=self.timeout)
            genome.fitnesses = fits
            genome.outputs = outputs
            genome.is_normal = is_normal
            genome.violate_num = violate_num
