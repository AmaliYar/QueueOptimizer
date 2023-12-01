import time
import numpy as np
from optimitzer.model import PlacementProcedure
from optimitzer.model import BRKGA
import optimitzer.plot as plot
from optimitzer import deps, graph, Input
from optimitzer.Input import ThreadProprs, Threads, QueueProps


# inputs = dict(
#     p=[1500, 382, 244, 216, 87, 99, 377, 250, 167, 139, 133, 164, 105, 102, 159, 185, 255, 71, 170, 176, 119, 68, 68,
#        68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68],
#     q=[50, 82, 31, 19, 25, 41, 25, 82, 25, 41, 25, 25, 21, 21, 31, 16, 12, 32, 15, 29, 43, 10, 10, 10, 10, 10, 10, 10,
#        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
#     r=[500, 618, 232, 28, 247, 412, 185, 618, 185, 309, 124, 124, 155, 103, 232, 159, 29, 161, 75, 72, 108, 26, 26, 26,
#        26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26],
#     L=[7000, 1000],
#     W=[200, 1000],
#     H=[1500, 1000])

t = Threads()
t.add_thread(ThreadProprs(1500, 50, 500))
t.add_thread(ThreadProprs(382, 82, 618))
t.add_thread(ThreadProprs(244, 31, 232))
t.add_thread(ThreadProprs(216, 19, 28))
t.add_thread(ThreadProprs(87, 25, 247))
t.add_thread(ThreadProprs(99, 41, 412))
t.add_thread(ThreadProprs(377, 25, 185))
t.add_thread(ThreadProprs(250, 82, 618))
t.add_thread(ThreadProprs(167, 25, 185))
t.add_thread(ThreadProprs(139, 41, 309))
t.add_thread(ThreadProprs(133, 25, 124))
t.add_thread(ThreadProprs(164, 25, 124))
t.add_thread(ThreadProprs(105, 21, 155))
t.add_thread(ThreadProprs(102, 21, 103))
t.add_thread(ThreadProprs(159, 31, 232))
t.add_thread(ThreadProprs(185, 16, 159))
t.add_thread(ThreadProprs(255, 12, 29))
t.add_thread(ThreadProprs(71, 32, 161))
t.add_thread(ThreadProprs(170, 15, 75))
t.add_thread(ThreadProprs(176, 29, 72))
t.add_thread(ThreadProprs(119, 43, 108))
t.add_thread(ThreadProprs(68, 10, 26))
t.add_thread(ThreadProprs(68, 10, 26))
t.add_thread(ThreadProprs(68, 10, 26))
t.add_thread(ThreadProprs(68, 10, 26))
t.add_thread(ThreadProprs(68, 10, 26))
t.add_thread(ThreadProprs(68, 10, 26))
t.add_thread(ThreadProprs(68, 10, 26))
t.add_thread(ThreadProprs(68, 10, 26))
t.add_thread(ThreadProprs(68, 10, 26))
t.add_thread(ThreadProprs(68, 10, 26))
t.add_thread(ThreadProprs(68, 10, 26))
t.add_thread(ThreadProprs(68, 10, 26))
t.add_thread(ThreadProprs(68, 10, 26))
t.add_thread(ThreadProprs(68, 10, 26))
t.add_thread(ThreadProprs(68, 10, 26))
t.add_thread(ThreadProprs(68, 10, 26))
t.add_thread(ThreadProprs(68, 10, 26))
t.add_thread(ThreadProprs(68, 10, 26))
t.add_thread(ThreadProprs(68, 10, 26))
t.add_thread(ThreadProprs(68, 10, 26))
t.add_thread(ThreadProprs(68, 10, 26))

t.add_thread(ThreadProprs(450, 120, 435))
t.add_thread(ThreadProprs(628, 79, 345))
t.add_thread(ThreadProprs(628, 79, 345))
t.add_thread(ThreadProprs(628, 79, 345))
t.add_thread(ThreadProprs(628, 79, 345))
t.add_thread(ThreadProprs(1200, 79, 345))
t.add_thread(ThreadProprs(1200, 79, 345))
q = QueueProps(t.total_time, 200, 1500)


# inputs = {'v': list(zip(inputs['p'], inputs['q'], inputs['r'])), 'V': list(zip(inputs['L'], inputs['W'], inputs['H']))}
# print('number of boxes:', len(inputs['v']))

start_time = time.time()

# inputs = generateInputs(75, 20, (600, 250, 250))

model = BRKGA(Input.pack_properties(t, q))
# model = BRKGA(inputs, num_generations=15, num_individuals=1000, num_elites=50, num_mutants=0, eliteCProb=0.7)

model.dependencies.add_dep(deps.Dependence(0, [1, 4, 5, 6, 7, 9, 11, 18, 19, 28, 30, 31]))
model.dependencies.add_dep(deps.Dependence(1, [2]))
model.dependencies.add_dep(deps.Dependence(2, [3]))
model.fit(patient=5, verbose=True)
print('used bins:', model.used_bins)
print('time:', time.time() - start_time)
# inputs['solution'] = model.solution
# decoder = PlacementProcedure(inputs, model.solution)
# decoder = PlacementProcedure(inputs, model.solution, model.dependencies)
decoder = PlacementProcedure(Input.pack_properties(t, q), model.solution, model.dependencies)
print('fitness:', decoder.evaluate())
print(np.argsort(model.solution[:11]))
# V = [(2500, 200, 1500), (1000, 200, 200)]
V = [(decoder.Bins[0].max_time*1.05, q.cpu, q.ram)]
print('est time {} secs'.format(decoder.Bins[0].max_time))
plot.Grapher(graph.generate_tree(decoder.boxes_placements)).draw()
plot.draw_3D_plots(decoder, V)
