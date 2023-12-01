import numpy as np

class Dependence:
    def __init__(self, codependent_id: int, dependents_id: list[int]):
        self.codep = codependent_id
        self.dep = dependents_id


class Dependencies:
    def __init__(self):
        self.deps: list = []

    def add_dep(self, dep: Dependence):
        self.deps.append(dep)

    def find_codependencies(self, dep_id) -> list:
        links: list = []
        for dep in self.deps:
            if dep_id in dep.dep:
                links.append(dep.codep)
        return links

    def eval_of_solutions(self, solutions) -> bool:
        conclusion: bool = False

        return conclusion


def validate_with_dependences(population, deps: Dependencies):
    for group in population:
        deps.eval_of_solutions(np.argsort(group[1][:11]))

