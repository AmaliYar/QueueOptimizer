class Box:
    def __init__(self, num: int, coords, box_num: int):
        self.id = num
        self.coords = coords #hight point - low point
        self.box_num = box_num


    def check_crossing(self, candidate, dimension_index) -> bool:
        crossing: bool = False
        dimensions = [0, 1, 2]
        dimensions.pop(dimension_index)
        if self.coords[1][dimension_index] == candidate.coords[0][dimension_index]:
            for dimention in dimensions:
                if (candidate.coords[0][dimention] > self.coords[1][dimention]) or (candidate.coords[0][dimention] <
                                                                                    self.coords[0][dimention]):
                    crossing = True
        return crossing
