from optimitzer import model
from optimitzer import deps
from optimitzer import plot
from optimitzer import graph
import numpy as np
from optimitzer.model import PlacementProcedure
import time
inputs = dict(
    p=[1500, 382, 244, 216, 87, 99, 377, 250, 167, 139, 133, 164, 105, 102, 159, 185, 255, 71, 170, 176, 119, 68, 68,
       68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68],
    q=[50, 82, 31, 19, 25, 41, 25, 82, 25, 41, 25, 25, 21, 21, 31, 16, 12, 32, 15, 29, 43, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    r=[500, 618, 232, 28, 247, 412, 185, 618, 185, 309, 124, 124, 155, 103, 232, 159, 29, 161, 75, 72, 108, 26, 26, 26,
       26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26],
    L=[7000, 1000],
    W=[200, 1000],
    H=[1500, 1000])

inputs = {'v': list(zip(inputs['p'], inputs['q'], inputs['r'])), 'V': list(zip(inputs['L'], inputs['W'], inputs['H']))}
print('number of boxes:', len(inputs['v']))

start_time = time.time()

# inputs = generateInputs(75, 20, (600, 250, 250))

model = model.BRKGA(inputs, num_generations=15, num_individuals=1000, num_elites=50, num_mutants=0, eliteCProb=0.7)

model.dependencies.add_dep(deps.Dependence(0, [1, 4, 5, 6, 7, 9, 11, 18, 19, 28, 30, 31]))
model.dependencies.add_dep(deps.Dependence(1, [2]))
model.dependencies.add_dep(deps.Dependence(2, [3]))
model.fit(patient=4, verbose=True)
print('used bins:', model.used_bins)
print('time:', time.time() - start_time)
inputs['solution'] = model.solution
# decoder = PlacementProcedure(inputs, model.solution)
decoder = PlacementProcedure(inputs, model.solution, model.dependencies)
print('fitness:', decoder.evaluate())
print(np.argsort(model.solution[:11]))
V = [(2500, 200, 1500), (1000, 200, 200)]

# def draw(decoder):
#     for i in range(decoder.num_opend_bins):
#         container = plot_3D(V=V[i])
#         for box in decoder.Bins[i].load_items:
#             container.add_box(box[0], box[1], mode='EMS')
#         print('Container', i, ':')
#         container.findOverlapping()
#         container.show()
# draw(decoder)
plot.Grapher(graph.generate_tree(decoder.boxes_placements)).draw()
plot.draw_3D_plots(decoder, V)
