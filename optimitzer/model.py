import math
import copy
import random
import numpy as np
import optimitzer.deps as deps
from optimitzer.box import Box
INFEASIBLE = 100000


def generateInstances(N = 20, m = 10, V = (100,100,100)):
    def ur(lb, ub):
        # u.r. is an abbreviation of "uniformly random". [Martello (1995)]
        value = random.uniform(lb, ub)
        return int(value) if value >= 1 else 1
    
    L, W, H = V
    p = []; q = []; r = []
    for i in range(N):
        p.append(ur(1/6*L, 1/4*L))
        q.append(ur(1/6*W, 1/4*W))
        r.append(ur(1/6*H, 1/4*H))
    
    L = [L]*m
    W = [W]*m
    H = [H]*m
    return range(N), range(m), p, q, r, L, W, H

def generateInputs(N, m, V):
    N, M, p,q,r, L,W,H =generateInstances(N, m, V)
    inputs = {'v':list(zip(p, q, r)), 'V':list(zip(L, W, H))}
    return inputs


# class Box:
#     def __init__(self, num: int, coords, box_num: int):
#         self.id = num
#         self.coords = coords #hight point - low point
#         self.box_num = box_num
#
#
#     def check_crossing(self, candidate, dimension_index) -> bool:
#         crossing: bool = False
#         dimensions = [0, 1, 2]
#         dimensions.pop(dimension_index)
#         if self.coords[1][dimension_index] == candidate.coords[0][dimension_index]:
#             for dimention in dimensions:
#                 if (candidate.coords[0][dimention] > self.coords[1][dimention]) or (candidate.coords[0][dimention] <
#                                                                                     self.coords[0][dimention]):
#                     crossing = True
#         return crossing


class Bin:
    def __init__(self, V, verbose=False):
        self.dimensions = V
        self.EMSs = [[np.array((0, 0, 0)), np.array(V)]]
        self.load_items = []

        if verbose:
            print('Init EMSs:',self.EMSs)
    
    def __getitem__(self, index):
        return self.EMSs[index]
    
    def __len__(self):
        return len(self.EMSs)
    
    def update(self, box, selected_EMS, min_vol = 1, min_dim = 1, verbose=False):

        # 1. place box in a EMS
        boxToPlace = np.array(box)
        selected_min = np.array(selected_EMS[0])
        ems = [selected_min, selected_min + boxToPlace]
        self.load_items.append(ems)
        if verbose:
            print('------------\n*Place Box*:\nEMS:', list(map(tuple, ems)))
        
        # 2. Generate new EMSs resulting from the intersection of the box
        for EMS in self.EMSs.copy():
            if self.overlapped(ems, EMS):
                
                # eliminate overlapped EMS
                self.eliminate(EMS)
                
                if verbose:
                    print('\n*Elimination*:\nRemove overlapped EMS:',
                          list(map(tuple, EMS)),
                          '\nEMSs left:',
                          list(map( lambda x : list(map(tuple,x)), self.EMSs)))
                
                # six new EMSs in 3 dimensionsc
                x1, y1, z1 = EMS[0]; x2, y2, z2 = EMS[1]
                x3, y3, z3 = ems[0]; x4, y4, z4 = ems[1]
                new_EMSs = [
                    [np.array((x4, y1, z1)), np.array((x2, y2, z2))],
                    [np.array((x1, y4, z1)), np.array((x2, y2, z2))],
                    [np.array((x1, y1, z4)), np.array((x2, y2, z2))]
                ]
                

                for new_EMS in new_EMSs:
                    new_box = new_EMS[1] - new_EMS[0]
                    isValid = True
                    
                    if verbose:
                        print('\n*New*\nEMS:', list(map(tuple, new_EMS)))

                    # 3. Eliminate new EMSs which are totally inscribed by other EMSs
                    for other_EMS in self.EMSs:
                        if self.inscribed(new_EMS, other_EMS):
                            isValid = False
                            if verbose:
                                print('-> Totally inscribed by:', list(map(tuple, other_EMS)))
                            
                    # 4. Do not add new EMS smaller than the volume of remaining boxes
                    if np.min(new_box) < min_dim:
                        isValid = False
                        if verbose:
                            print('-> Dimension too small.')
                        
                    # 5. Do not add new EMS having smaller dimension of the smallest dimension of remaining boxes
                    if np.product(new_box) < min_vol:
                        isValid = False
                        if verbose:
                            print('-> Volumne too small.')

                    if isValid:
                        self.EMSs.append(new_EMS)
                        if verbose:
                            print('-> Success\nAdd new EMS:', list(map(tuple, new_EMS)))

        if verbose:
            print('\nEnd:')
            print('EMSs:', list(map( lambda x : list(map(tuple,x)), self.EMSs)))
    
    def overlapped(self, ems, EMS):
        if np.all(ems[1] > EMS[0]) and np.all(ems[0] < EMS[1]):
            return True
        return False
    
    def inscribed(self, ems, EMS):
        if np.all(EMS[0] <= ems[0]) and np.all(ems[1] <= EMS[1]):
            return True
        return False
    
    def eliminate(self, ems):
        # numpy array can't compare directly
        ems = list(map(tuple, ems))    
        for index, EMS in enumerate(self.EMSs):
            if ems == list(map(tuple, EMS)):
                self.EMSs.pop(index)
                return
    
    def get_EMSs(self):
        return  list(map( lambda x : list(map(tuple,x)), self.EMSs))
    
    def load(self):
        return np.sum([ np.product(item[1] - item[0]) for item in self.load_items]) / np.product(self.dimensions)
    
class PlacementProcedure():
    # def __init__(self, inputs, solution, verbose=False):
    def __init__(self, inputs, solution, deps: deps.Dependencies, verbose=False):
        self.Bins = [Bin(V) for V in inputs['V']]
        self.boxes = inputs['v']
        self.BPS = np.argsort(solution[:len(self.boxes)])
        self.VBO = solution[len(self.boxes):]
        self.num_opend_bins = 1
        # self.directions = [1,2,3,4,5,6]
        self.directions = [1]
        self.crossing = False
        self.verbose = verbose
        if self.verbose:
            print('------------------------------------------------------------------')
            print('|   Placement Procedure')
            print('|    -> Boxes:', self.boxes)
            print('|    -> Box Packing Sequence:', self.BPS)
            print('|    -> Vector of Box Orientations:', self.VBO)
            print('-------------------------------------------------------------------')
        self.boxes_placements: list[Box] = []
        self.deps = deps
        self.infisible = False
        self.placement()

    
    def placement(self):
        items_sorted = [self.boxes[i] for i in self.BPS]

        # Box Selection
        for i, box in enumerate(items_sorted):
            if self.verbose:
                print('Select Box:', box)
            # Bin and EMS selection
            selected_bin = None
            selected_EMS = None
            EMS = None
            for k in range(self.num_opend_bins):
                # select EMS using DFTRC-2
                # EMS = self.DFTRC_2(box, k)
                if self.check_codep_box_nums(self.deps.find_codependencies(self.BPS[i]), k):
                    # EMS = self.DFTRC_2(box, k)
                    EMS = self.DFTRC_2(box, k, self.BPS[i])

                # update selection if "packable"
                if EMS != None:
                    selected_bin = k
                    selected_EMS = EMS
                    break
            
            # Open new empty bin
            if selected_bin == None:
                self.num_opend_bins += 1
                selected_bin = self.num_opend_bins - 1
                if self.num_opend_bins > len(self.Bins):
                    self.infisible = True
                    
                    if self.verbose:
                        print('No more bin to open. [Infeasible]')
                    return
                    
                selected_EMS = self.Bins[selected_bin].EMSs[0] # origin of the new bin
                if self.verbose:
                    print('No available bin... open bin', selected_bin)
            
            if self.verbose:
                print('Select EMS:', list(map(tuple, selected_EMS)))
                
            # Box orientation selection
            # BO = self.selecte_box_orientaion(self.VBO[i], box, selected_EMS)
            try:
                BO = self.selecte_box_orientaion(self.VBO[i], box, selected_EMS)
            except:
                print('hast place for box {}'.format(i))

            # elimination rule for different process
            min_vol, min_dim = self.elimination_rule(items_sorted[i+1:])
            
            self.boxes_placements.append(Box(self.BPS[i], [np.array(self.orient(box, BO))+selected_EMS[0], selected_EMS[0]], selected_bin))

            # pack the box to the bin & update state information
            self.Bins[selected_bin].update(self.orient(box, BO), selected_EMS, min_vol, min_dim)
            if self.verbose:
                print('Add box to Bin', selected_bin)
                print(' -> EMSs:',self.Bins[selected_bin].get_EMSs())
                print('------------------------------------------------------------')
        if self.verbose:
            print('|')
            print('|     Number of used bins:', self.num_opend_bins)
            print('|')
            print('------------------------------------------------------------')

    # Distance to the Front-Top-Right Corner
    # def DFTRC_2(self, box, k):
    def DFTRC_2(self, box, k, box_id):
        maxDist = -1
        selectedEMS = None
        updated_EMSs = self.get_all_EMSs(self.deps.find_codependencies(box_id),
                                         self.Bins[k].EMSs)

        for EMS in updated_EMSs:

        # for EMS in self.Bins[k].EMSs:
            D, W, H = self.Bins[k].dimensions
            for direction in self.directions:
                d, w, h = self.orient(box, direction)
                if self.fitin((d, w, h), EMS):
                    x, y, z = EMS[0]
                    distance = pow(D-x-d, 2) + pow(W-y-w, 2) + pow(H-z-h, 2)

                    if distance > maxDist:
                        maxDist = distance
                        selectedEMS = EMS
        return selectedEMS

    def orient(self, box, BO=1):
        d, w, h = box
        if BO == 1: return (d, w, h)
        elif BO == 2: return (d, h, w)
        elif BO == 3: return (w, d, h)
        elif BO == 4: return (w, h, d)
        elif BO == 5: return (h, d, w)
        elif BO == 6: return (h, w, d)
        
    #todo: add exception for case if box if full
    def selecte_box_orientaion(self, VBO, box, EMS):
        # feasible direction
        BOs = []
        for direction in self.directions:
            if self.fitin(self.orient(box, direction), EMS):
                BOs.append(direction)
        # choose direction based on VBO vector

        selectedBO = BOs[math.ceil(VBO*len(BOs))-1]

        if self.verbose:
            print('Select VBO:', selectedBO,'  (BOs',BOs, ', vector', VBO,')')
        return selectedBO
    
    def fitin(self, box, EMS):
        # all dimension fit
        for d in range(3):
            if box[d] > EMS[1][d] - EMS[0][d]:
                return False
        return True
    
    def elimination_rule(self, remaining_boxes):
        if len(remaining_boxes) == 0:
            return 0, 0
        
        min_vol = 999999999
        min_dim = 9999
        for box in remaining_boxes:
            # minimum dimension
            dim = np.min(box)
            if dim < min_dim:
                min_dim = dim
                
            # minimum volume
            vol = np.product(box)
            if vol < min_vol:
                min_vol = vol
        return min_vol, min_dim
    
    def evaluate(self):
        if self.infisible:
            return INFEASIBLE
        
        leastLoad = 1
        for k in range(self.num_opend_bins):
            load = self.Bins[k].load()
            if load < leastLoad:
                leastLoad = load
        return self.num_opend_bins + leastLoad%1

    def check_dependencies(self, depends_list: deps.Dependencies):
        self.crossing = False
        for box in self.BPS:
            if depends_list.find_codependencies(box):
                if self.compare_coords(box, depends_list.find_codependencies(box), 0):
                    self.crossing = True

    def compare_coords(self, dependence_id: int, codependencies: list, comparing_coord_index):
        crossing = False
        dependence_props = self.find_box_and_coords(dependence_id)
        for codependency_id in codependencies:
            codep_proprs = self.find_box_and_coords(codependency_id)
            try: self.check_boxes_positions(codep_proprs[0],
                                          dependence_props[0],
                                          codep_proprs[1][1][comparing_coord_index],
                                          dependence_props[1][0][comparing_coord_index])
            except:
                crossing = True
                break
        return crossing
    def find_box_and_coords(self, box_num):
        iterator = 0
        for bin in self.Bins:
            iterator += (len(bin.load_items))
            if box_num <= (iterator - 1):
                return [self.Bins.index(bin),
                        bin.load_items[(len(bin.load_items) - 1) - (iterator - box_num)]]

    def check_boxes_positions(self, prev_box_pos, next_box_pos, prev_end, next_start) -> bool:
        is_incorrect = False
        if prev_box_pos > next_box_pos:
            is_incorrect = True
        if prev_end > next_start:
            is_incorrect = True
        return is_incorrect

    #method to check numbers of boxes with codependecies
    def check_codep_box_nums(self, codeps_list: list[int], current_box_num: int) -> bool:
        avalible: bool = True
        for codep in codeps_list:
            if self.get_box(codep).box_num > current_box_num:
                avalible = False
        return avalible

    def get_box(self, box_num) -> Box:
        for box in self.boxes_placements:
            if box.id == box_num:
                return box

    def get_avalible_EMSs(self, codep_top_coords, EMSs: list, coordinate_index=0):
        avalible_EMSs: list = []
        for EMS in EMSs:
            if (EMS[0] - codep_top_coords)[coordinate_index] >= 0:
                avalible_EMSs.append(EMS)
        return avalible_EMSs

    def get_all_EMSs(self, codeps_list: list, EMSs: list, coordinate_index = 0):
        total_list: list = EMSs
        for codep in codeps_list:
            codep_box = self.get_box(codep)
            # total_list = list(set(total_list).intersection(self.get_avalible_EMSs(codep_box.coords[0], EMSs, coordinate_index)))
            total_list = inner_join_composite_list(total_list, self.get_avalible_EMSs(codep_box.coords[0], EMSs, coordinate_index))
        return total_list


class BRKGA():
    def __init__(self, inputs, num_generations = 200, num_individuals=120, num_elites = 12, num_mutants = 18, eliteCProb = 0.7, multiProcess = False):
        # Setting
        self.multiProcess = multiProcess
        # Input
        self.inputs = copy.deepcopy(inputs)
        self.N = len(inputs['v'])
        
        # Configuration
        self.num_generations = num_generations
        self.num_individuals = int(num_individuals)
        self.num_gene = 2*self.N
        
        self.num_elites = int(num_elites)
        self.num_mutants = int(num_mutants)
        self.eliteCProb = eliteCProb
        
        # Result
        self.used_bins = -1
        self.solution = None
        self.best_fitness = -1
        self.history = {
            'mean': [],
            'min': []
        }
        self.dependencies = deps.Dependencies()
        
    def decoder(self, solution):
        # placement = PlacementProcedure(self.inputs, solution)
        placement = PlacementProcedure(self.inputs, solution, self.dependencies)
        return placement.evaluate()
    
    def cal_fitness(self, population):
        fitness_list = list()
        dependencies_corrections = list()
        for solution in population:
            # decoder = PlacementProcedure(self.inputs, solution)
            decoder = PlacementProcedure(self.inputs, solution, self.dependencies)
            decoder.check_dependencies(self.dependencies)
            dependencies_corrections.append(decoder.crossing)
            fitness_list.append(decoder.evaluate())
        return [fitness_list, dependencies_corrections]

    def partition(self, population, fitness_list):
        sorted_indexs = np.argsort(fitness_list)
        return population[sorted_indexs[:self.num_elites]], population[sorted_indexs[self.num_elites:]], \
        np.array(fitness_list)[sorted_indexs[:self.num_elites]]
        # return population[sorted_indexs[:self.num_elites]], population[sorted_indexs[self.num_elites:]], fitness_list[sorted_indexs[:self.num_elites]]

    def crossover(self, elite, non_elite):
        # chance to choose the gene from elite and non_elite for each gene
        return [elite[gene] if np.random.uniform(low=0.0, high=1.0) < self.eliteCProb else non_elite[gene] for gene in range(self.num_gene)]
    
    def mating(self, elites, non_elites):
        # biased selection of mating parents: 1 elite & 1 non_elite
        num_offspring = self.num_individuals - self.num_elites - self.num_mutants
        return [self.crossover(random.choice(elites), random.choice(non_elites)) for i in range(num_offspring)]
    
    def mutants(self):
        return np.random.uniform(low=0.0, high=1.0, size=(self.num_mutants, self.num_gene))

    def generate_population(self):
        population = np.array([])
        while len(population) < self.num_individuals:
            population_iteration = np.random.uniform(low=0.0, high=1.0, size=(self.num_individuals, self.num_gene))
            if population.size == 0:
                population = self.check_deps_compliance(population_iteration)
            else:
                clear = self.check_deps_compliance(population_iteration)
                if clear.size > 0:
                    population = np.append(population, clear, 0)
        return population[:self.num_individuals]



    def fit(self, patient = 10, verbose = False):
        # Initial population & fitness
        # population = np.random.uniform(low=0.0, high=1.0, size=(self.num_individuals, self.num_gene))
        # population = self.check_deps_compliance(population)
        population = self.generate_population()
        fitness_list, dependencies_rule = self.cal_fitness(population)
        
        if verbose:
            print('\nInitial Population:')
            print('  ->  shape:', population.shape)
            print('  ->  Best Fitness:', max(fitness_list))
            
        # best
        # best_fitness = np.min(fitness_list)
        best_fitness = np.max(fitness_list)
        # best_solution = population[np.argmin(fitness_list)]
        best_solution = population[np.argmax(fitness_list)]
        index = 0
        for fitness in fitness_list:
            if not dependencies_rule[index]:
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = population[index]


        self.history['min'].append(np.min(fitness_list))
        self.history['mean'].append(np.mean(fitness_list))
        
        
        # Repeat generations
        best_iter = 0
        for g in range(self.num_generations):
            # early stopping
            if g - best_iter > patient:
                self.used_bins = math.floor(best_fitness)
                self.best_fitness = best_fitness
                self.solution = best_solution
                if verbose:
                    print('Early stop at iter', g, '(timeout)')
                return 'feasible'
            
            # Select elite group
            elites, non_elites, elite_fitness_list = self.partition(population, fitness_list)
            
            # Biased Mating & Crossover
            offsprings = self.mating(elites, non_elites)
            
            # Generate mutants
            mutants = self.mutants()

            # New Population & fitness
            offspring = np.concatenate((mutants, offsprings), axis=0)
            # offspring_fitness_list = self.cal_fitness(offspring)
            
            # population = np.concatenate((elites, offsprings), axis = 0)
            population = np.concatenate((elites, mutants, offsprings), axis=0)
            # fitness_list = elite_fitness_list + offspring_fitness_list

            population = self.check_deps_compliance(population)
            fitness_list, dependencies_rule = self.cal_fitness(population)

            
            # Update Best Fitness
            #np.argsort(population[1][:11])
            index = 0
            for fitness in fitness_list:
                if fitness < best_fitness:
                    best_iter = g
                    best_fitness = fitness
                    if not dependencies_rule[index]:
                        best_solution = population[np.argmin(fitness_list)]
                index += 1
            self.history['min'].append(np.min(fitness_list))
            self.history['mean'].append(np.mean(fitness_list))
            
            if verbose:
                print("Generation :", g, ' \t(Best Fitness:', best_fitness,')')
            
        self.used_bins = math.floor(best_fitness)
        self.best_fitness = best_fitness
        self.solution = best_solution
        return 'feasible'
    def check_deps_compliance(self, population):
        updated = []
        for config in population:
            complied = True
            boxes_list = np.argsort(config[:self.N])
            for box in boxes_list:
                for codep in self.dependencies.find_codependencies(box):
                    if int(list(np.where(boxes_list == box))[0]) < int(list(np.where(boxes_list == codep))[0]):
                        complied = False
                        break
            if complied:
                updated.append(config)
                # print(np.argsort(config[:self.N]))
        # if len(updated)>0: print('after genocide left {}'.format(len(updated)))
        return np.array(updated)


def inner_join_composite_list(common_list, local_list):
    joined_list: list = []
    for element in common_list:
        for local_element in local_list:
            if compare_3d_2points_objects(element, local_element):
                joined_list.append(element)
    return joined_list


def compare_3d_2points_objects(obj1, obj2):
    equal = False
    if (np.array(obj1) == np.array(obj2)).all():
        equal = True
    return equal

