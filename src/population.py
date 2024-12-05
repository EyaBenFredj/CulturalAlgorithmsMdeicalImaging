import random

class Individual:
    def __init__(self, features=None, preprocessing=None):

        self.features = features if features is not None else ["feature1", "feature2"]  # Example features
        self.preprocessing = preprocessing if preprocessing is not None else ["none"]  # Example preprocessing methods
        self.classification_threshold = random.uniform(0.3, 0.9)  # Random initial threshold

    def copy(self):

        new_individual = Individual()
        new_individual.features = self.features[:]
        new_individual.preprocessing = self.preprocessing[:]
        new_individual.classification_threshold = self.classification_threshold
        return new_individual

    def mutate(self, mutation_rate=0.1):
        if random.random() < mutation_rate:
            # Example mutation logic for features (replace a feature)
            if self.features:
                index = random.randint(0, len(self.features) - 1)
                self.features[index] = f"mutated_feature_{random.randint(1, 100)}"

        if random.random() < mutation_rate:
            # Example mutation logic for preprocessing (add/change preprocessing step)
            self.preprocessing = [f"mutated_preprocessing_{random.randint(1, 100)}"]

        if random.random() < mutation_rate:
            # Mutate the classification threshold
            self.classification_threshold = random.uniform(0.3, 0.9)


class Population:
    def __init__(self, initial_size):
        self.individuals = [Individual() for _ in range(initial_size)]

    def evaluate_fitness(self, fitness_function):
        fitness_scores = [fitness_function(individual) for individual in self.individuals]
        return fitness_scores

    def select(self, num_to_select):
        return random.sample(self.individuals, num_to_select)

    def reproduce(self, parent1, parent2):
        child = Individual()
        child.features = random.choice([parent1.features, parent2.features])[:]  # Choose features from one parent
        child.preprocessing = random.choice([parent1.preprocessing, parent2.preprocessing])[:]  # Choose preprocessing
        child.classification_threshold = (parent1.classification_threshold + parent2.classification_threshold) / 2
        return child

    def generate_next_generation(self, mutation_rate=0.1):
        new_individuals = []
        num_to_select = len(self.individuals) // 2  # Select half the population size for reproduction
        selected_individuals = self.select(num_to_select)

        # Create offspring
        for i in range(len(self.individuals)):
            parent1, parent2 = random.sample(selected_individuals, 2)
            child = self.reproduce(parent1, parent2)
            child.mutate(mutation_rate)
            new_individuals.append(child)

        self.individuals = new_individuals  # Replace old population with new generation
