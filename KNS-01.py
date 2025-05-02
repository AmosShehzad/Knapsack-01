import random

class Chromosome:    
    def __init__(self, genes, knapsack, weight_limit):
        self.genes = list(genes)
        self.knapsack = knapsack
        self.weight_limit = weight_limit
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        total_value = 0
        total_weight = 0
        for gene, (value, weight) in zip(self.genes, self.knapsack.values()):
            if gene == 1:
                total_value += value
                total_weight += weight
        if total_weight > self.weight_limit:
            return 0
        return total_value

    def __str__(self):
        return f"Genes: {self.genes}, Fitness: {self.fitness}"


class GeneticAlgorithm:
    def __init__(self, weight_limit, knapsack, population_size, mutation_rate):
        self.weight_limit = weight_limit
        self.knapsack = knapsack
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for j in range(self.population_size):
            genes = [random.randint(0, 1) for j in range(len(self.knapsack))]
            population.append(Chromosome(genes, self.knapsack, self.weight_limit))
        return population

    def selection(self):
        # Elitism: Keep the best individual
        sorted_pop = sorted(self.population, key=lambda c: c.fitness, reverse=True)
        selected = [sorted_pop[0]]  # Elitism
        
        # Roulette Wheel Selection
        fitness_sum = sum(c.fitness for c in self.population)
        if fitness_sum == 0:
            return random.sample(self.population, self.population_size)

        probabilities = [c.fitness / fitness_sum for c in self.population]
        selected += random.choices(self.population, weights=probabilities, k=self.population_size - 1)
        return selected

    def crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1.genes) - 1)
        child1_genes = parent1.genes[:point] + parent2.genes[point:]
        child2_genes = parent2.genes[:point] + parent1.genes[point:]
        return (Chromosome(child1_genes, self.knapsack, self.weight_limit),
                Chromosome(child2_genes, self.knapsack, self.weight_limit))

    def mutation(self, chromosome):
        for i in range(len(chromosome.genes)):
            if random.random() < self.mutation_rate:
                chromosome.genes[i] = 1 - chromosome.genes[i]
        chromosome.fitness = chromosome.calculate_fitness()

    def evolve(self):
        new_population = []
        selected = self.selection()
        for i in range(0, self.population_size, 2):
            parent1 = selected[i]
            parent2 = selected[(i + 1) % self.population_size]
            child1, child2 = self.crossover(parent1, parent2)
            self.mutation(child1)
            self.mutation(child2)
            new_population.extend([child1, child2])
        self.population = new_population[:self.population_size]

    def get_solution(self):
        return max(self.population, key=lambda c: c.fitness)


def build_knapsack(file):
    knapsack = {}
    with open(file, 'r') as f:
        lines = f.readlines()
        w,weight_limit = map(int, lines[0].split())
        for idx, line in enumerate(lines[1:], start=0):
            value, weight = map(int, line.split())
            knapsack[idx] = (value, weight)
    return weight_limit, knapsack


if __name__ == "__main__":
    w, knapsack = build_knapsack("test.txt")
    ga = GeneticAlgorithm(weight_limit=w, knapsack=knapsack, population_size=10, mutation_rate=0.1)

    for _ in range(50):
        ga.evolve()

    best_solution = ga.get_solution()
    print("Best solution found:", best_solution)
    print("Fitness of best solution:", best_solution.fitness)
