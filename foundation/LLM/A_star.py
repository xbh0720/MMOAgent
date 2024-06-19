from math import inf

import numpy as np
import sortedcontainers


class Point:
    def __init__(self, x, y, parent=None, g_score=inf, f_score=inf):
        self.x = x
        self.y = y
        self.g_score = g_score
        self.f_score = f_score
        self.parent = parent
        self.closed = False
        self.in_openset = False


class OpenSet:
    def __init__(self) -> None:
        self.sortedlist = sortedcontainers.SortedList(key=lambda x: x.f_score)

    def push(self, item) -> None:
        item.in_openset = True
        self.sortedlist.add(item)

    def pop(self):
        item = self.sortedlist.pop(0)
        item.in_openset = False
        return item

    def remove(self, item) -> None:
        self.sortedlist.remove(item)
        item.in_openset = False

    def __len__(self) -> int:
        return len(self.sortedlist)


class AStar:
    def __init__(self):
        
        self.nodes_list = {}

    def build_node(self, x, y):
        loc = (x, y)
        if loc not in self.nodes_list.keys():
            self.nodes_list[loc] = Point(x, y)
        return self.nodes_list[loc]

    def distance_between(self, p, p_neighbor):
        return 1  

    def HeuristicCost(self, current_p, goal_p):
        x1, y1 = current_p.x, current_p.y
        x2, y2 = goal_p.x, goal_p.y
        return abs(x2 - x1) + abs(y2 - y1)

    def is_obstacle(self, x, y):
        if self.road_map[x][y] < 0:
            return True
        return False

    def IsValidPoint(self, x, y):
        if x < 0 or y < 0:
            return False
        if x >= self.height or y >= self.width:
            return False
        return not self.is_obstacle(x, y) 

    def IsEndPoint(self, p, goal):
        # self.destination,
        return p.x == goal.x and p.y == goal.y

    def neighbor_points(self, p):
        relative_locs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        x, y = p.x, p.y
        neighbors = []
        for relative_loc in relative_locs:
            relative_x, relative_y = relative_loc
            x_neighbor, y_neighbor = x + relative_x, y + relative_y
            if self.IsValidPoint(x_neighbor, y_neighbor):
                neighbors.append(self.build_node(x_neighbor, y_neighbor))
        return neighbors

    
    def astar(self, start, goal, map):
        self.road_map = map
        self.height = np.shape(map)[0]
        self.width = np.shape(map)[1]
        self.nodes_list = {}
        start_p = Point(start[0], start[1], g_score=0)
        if isinstance(goal, (list, tuple)):
            pass
        elif isinstance(goal, str):
            return []
        goal = Point(goal[0], goal[1])
        start_p.f_score = self.HeuristicCost(start_p, goal)
        if self.IsEndPoint(start_p, goal):
            return [(start_p.x, start_p.y)]

        openSet = OpenSet()

        openSet.push(start_p)

        while openSet:
            current = openSet.pop()

            if self.IsEndPoint(current, goal):
                return self.BuildPath(current)

            current.closed = True

            neighbors = self.neighbor_points(current)
            for neighbor in neighbors:
                if neighbor.closed:
                    continue

                tentative_gscore = current.g_score + self.distance_between(
                    current, neighbor
                )

                if tentative_gscore >= neighbor.g_score:
                    continue

                neighbor_from_openset = neighbor.in_openset

                if neighbor_from_openset:
                    # we have to remove the item from the heap, as its score has changed
                    openSet.remove(neighbor)

                # update the node
                neighbor.parent = current
                neighbor.g_score = tentative_gscore
                neighbor.f_score = tentative_gscore + self.HeuristicCost(neighbor, goal)

                openSet.push(neighbor)

        return []

    def BuildPath(self, last_p):
        path = []
        p = last_p
        while p:
            path.insert(0, (p.x, p.y))  # Insert first
            p = p.parent
        
        return path
