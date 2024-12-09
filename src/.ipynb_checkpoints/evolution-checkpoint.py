import random
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Evolution:
    def __init__(self, population, belief_space, elitism_count=2, verbose=True):
        self.population = population
        self.belief_space = belief_space
        self.elitism_count = elitism_count
        self.verbose = verbose
        if self.verbose:
            logging.info("Evolution initialized with population size: %d", len(self.population.individuals))

    def select_parents(self) -> Tuple:
        tournament_size = max(2, len(self.population.individuals) // 10)  # Dynamic tournament size
        parent1 = self._tournament_selection(tournament_size)
        parent2 = self._tournament_selection(tournament_size)
        if self.verbose:
            logging.debug("Selected parents: %s and %s", parent1.id, parent2.id)
        return parent1, parent2

    def _tournament_selection(self, tournament_size: int):
        candidates = random.sample(self.population.individuals, tournament_size)
        selected = max(candidates, key=lambda ind: ind.fitness)
        return selected

    def evaluate_fitness(self):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._compute_fitness, self.population.individuals))

        for individual, fitness in zip(self.population.individuals, results):
            individual.fitness = fitness
            if self.verbose:
                logging.debug("Evaluated fitness for individual %s: %f", individual.id, individual.fitness)

    def _compute_fitness(self, individual):
        accuracy = individual.model.evaluate_accuracy()
        sensitivity = individual.model.evaluate_sensitivity()
        specificity = individual.model.evaluate_specificity()
        fitness = 0.5 * accuracy + 0.3 * sensitivity + 0.2 * specificity
        return fitness

    def crossover(self, parent1, parent2):
        offspring = parent1.copy()  # Shallow copy to optimize
        offspring.features = [
            gene1 if random.random() < 0.5 else gene2
            for gene1, gene2 in zip(parent1.features, parent2.features)
        ]
        offspring.preprocessing = [
            step1 if random.random() < 0.5 else step2
            for step1, step2 in zip(parent1.preprocessing, parent2.preprocessing)
        ]
        offspring.classification_threshold = (
            0.6 * parent1.classification_threshold +
            0.4 * parent2.classification_threshold
        )
        offspring.id = self.population.generate_unique_id()
        return offspring

    def mutate(self, individual, mutation_rate=0.05):
        if random.random() < mutation_rate:
            individual.features[random.randint(0, len(individual.features) - 1)] = "mutated_feature"
        return individual

    def evolve_population(self):
        self.evaluate_fitness()
        sorted_individuals = sorted(self.population.individuals, key=lambda ind: ind.fitness, reverse=True)
        elites = sorted_individuals[:self.elitism_count]
        new_population = elites

        while len(new_population) < len(self.population.individuals):
            parent1, parent2 = self.select_parents()
            offspring = self.crossover(parent1, parent2)
            offspring = self.mutate(offspring)
            new_population.append(offspring)

        self.population.individuals = new_population
        self.belief_space.update(self.population.individuals)
        return self.population

    def get_best_individual(self):
        best_individual = max(self.population.individuals, key=lambda ind: ind.fitness)
        return best_individual
