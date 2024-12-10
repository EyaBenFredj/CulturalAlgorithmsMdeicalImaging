import random
class BeliefSpace:

    def __init__(self):
       
        self.normative_knowledge = []
        self.situational_knowledge = []
        self.domain_specific_knowledge = []
        self.statistical_knowledge = []

    def update(self, individual, performance):

        self.normative_knowledge.append({
            "features": individual.features,
            "preprocessing": individual.preprocessing,
            "threshold": individual.classification_threshold,
            "performance": performance
        })

  
        self.prune_knowledge()

    def prune_knowledge(self):
        """
        Prune outdated or less relevant knowledge from the belief space.
        This is a simple example where we retain only the top-performing entries.
        """
       
        if len(self.normative_knowledge) > 10:
            self.normative_knowledge.sort(key=lambda x: x["performance"], reverse=True)
            self.normative_knowledge = self.normative_knowledge[:10]



    def guide(self, population):
        if not self.normative_knowledge:
            return

        best_knowledge = max(self.normative_knowledge, key=lambda x: x["performance"])

        for individual in population:
            if random.random() < 0.5:  # 50% chance to be influenced
                individual.features = best_knowledge["features"][:]
                individual.preprocessing = best_knowledge["preprocessing"][:]
                individual.classification_threshold = best_knowledge["threshold"]
