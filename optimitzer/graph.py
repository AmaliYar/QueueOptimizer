import optimitzer.box as box
class Node:
    def __init__(self, index, parent=None):
        self.index = index
        self.parent_index = parent
        self.wide = 1
        self.childs = []
        self.top = False
        if parent is None:
            self.top = True

    def add_child(self, child):
        self.childs.append(child)
        self.calc_level_wide()

    def calc_level_wide(self):
        if len(self.childs) > self.wide:
            self.wide = len(self.childs)

    def find_parent(self, parent_id, new_child):
        if parent_id == self.index:
            self.add_child(new_child)
            pass
        if self.childs:
            for child in self.childs:
                if parent_id == child.index:
                    child.add_child(new_child)
                    break
                else:
                    child.find_parent(parent_id, new_child)

    def calculate_wide(self, wide):
        if len(self.childs) > wide:
            wide = self.childs
        if self.childs:
            self.calculate_wide(wide)


class Thread:
    def __init__(self):
        self.subtree: list = []
        self.nodes: int = 0

    def add_node(self, node: Node):
        self.subtree.append(node)

    def put_child_node(self, parent_list: list, child_node: Node):
        for parent in parent_list:
            for node in self.subtree:
                node.find_parent(parent, child_node)
            # for node in list(
            #         filter(
            #             lambda thrd: parent in list(map(lambda d: d.index, thrd.childs)) or parent == thrd.index,
            #             self.subtree)):
            #     node.find_parent(parent, child_node)


def generate_tree(boxes_list):
    thread = Thread()
    thread.nodes = len(boxes_list)
    for box in sorted(boxes_list,
                      key=lambda q: list(q.coords)[1][0]):
        #catch first by X boxes (X = 0)
        if box.coords[1][0] == 0:
            thread.add_node(Node(box.id))
        else:
            # add dependency box for codependencies
            thread.put_child_node(
                find_prev_box(box, boxes_list),
                Node(box.id))
    return thread

# подаем координаты искомого ящика
# проходим все ящики, которые мы прошли ранее
# признаком предшественника будет равное значение по иксу (должны быть равны) окончание одного есть начало другого
# собираем предшесвтенников в лист
# определяем номера родителей
# добавляем через find parent или add child (если это дети первого уровня)
# /


def find_prev_box(current_box: box.Box, added_boxes: list[box.Box]) -> list:
    prevs: list = []
    for prev in added_boxes:
        if current_box.check_crossing(prev, 0):
            prevs.append(prev.id)
    # print(prevs, current_box.id)
    return prevs


def drop_far_boxes(boxes: list[box.Box], dimention_index: int = 0):
    return list(filter(
        lambda obj: obj.coords[0][dimention_index] == sorted(
            boxes,
            key=lambda q: q.coords[0][dimention_index])[0].coords[0][dimention_index],
        boxes))






