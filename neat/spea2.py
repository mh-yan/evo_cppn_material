# Update the fitness calculation to include strength calculation
def calculate_fitness(population, len_archive):
    # k = int((2 * len(population)) ** 0.5)
    k = 2
    distance_matrix = calculate_distance_matrix(population)
    for individual in population:
        individual.strength = calculate_strength(individual, population)
    for individual in population:
        individual.fitness = -(
            calculate_raw_fitness(individual, population)
            + calculate_density(individual, distance_matrix, k)
        )


def calculate_raw_fitness(individual, population):
    raw_fitness = 0
    for other_individual in population:
        if dominates(other_individual, individual):
            # Assuming strength is a property of the individual
            raw_fitness += other_individual.strength
    return raw_fitness


def calculate_strength(individual, population):
    # Assuming strength is the number of individuals that the current individual dominates
    return sum(
        1 for other_individual in population if dominates(individual, other_individual)
    )


def calculate_density(individual, distance_matrix, k):
    return 1 / (distance_matrix[individual][k] + 2)


def calculate_distance_matrix(population):

    distance_matrix = {}
    for individual in population:
        distances = [
            calculate_distance(individual, other_individual)
            for other_individual in population
        ]
        distances.sort()
        distance_matrix[individual] = distances
    return distance_matrix


def calculate_distance(individual1, individual2):
    return (
        sum(
            (individual1.fitnesses[i] - individual2.fitnesses[i]) ** 2
            for i in range(len(individual1.fitnesses))
        )
        ** 0.5
    )


def dominates(individual1, individual2):
    return all(
        f1 >= f2 for f1, f2 in zip(individual1.fitnesses, individual2.fitnesses)
    ) and any(f1 > f2 for f1, f2 in zip(individual1.fitnesses, individual2.fitnesses))


# Return all nondominated individuals in the population and those who dominate
def get_nondominated_and_dominating(population, archive):
    nondominated = []
    dominating = []
    combined = list(set(population + archive))  # remove duplicates
    print(f"len combined {len(combined)}")
    for individual in combined:
        if all(
            not dominates(other_individual, individual) for other_individual in combined
        ):
            nondominated.append(individual)
        else:
            dominating.append(individual)

    # Sort the dominated individuals by the number of individuals they dominate
    dominating.sort(
        key=lambda individual: calculate_strength(individual, combined), reverse=True
    )

    return nondominated, dominating


# Get the archive
def get_archive(archive, nondominated, dominated, archive_size):
    # Add nondominated solutions to the archive

    archive.extend(nondominated)
    # 重复的元素去掉
    archive = list(set(archive))

    # If the archive is not full, add from dominate
    if len(archive) < archive_size:
        additional_needed = archive_size - len(archive)
        archive.extend(dominated[:additional_needed])
        archive = list(set(archive))

    # If the archive is too large, prune it based on density
    elif len(archive) > archive_size:
        # Calculate the k value for density calculation
        k = 2

        # Calculate the distance matrix for each individual in the archive
        distance_matrix = calculate_distance_matrix(archive)

        # Sort the archive based on distance (larger is better)
        archive.sort(
            key=lambda individual: distance_matrix[individual][k], reverse=True
        )

        # Remove individuals with fitnesses[0] == -2
        archive = [
            individual for individual in archive if individual.fitnesses[0] != -2
        ]
        # Prune the archive to the desired size
        archive = archive[:archive_size]
    return archive
