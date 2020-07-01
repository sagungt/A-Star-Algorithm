import tkinter as tk
import heapq
import numpy as np
import time
from functools import partial


class Field():
    def __init__(self):
        self.window = tk.Tk()
        self.window.geometry("{0}x{1}+0+0".format(
            self.window.winfo_screenwidth()-3, self.window.winfo_screenheight()-3))
        self.x = 28
        self.y = 33
        self.start = [0, 0]
        self.goal = [32, 27]
        self.grid = np.zeros((self.y, self.x), dtype=np.int32)
        self.__init()

    def __init(self):
        self.button_frame = tk.Frame(self.window)
        self.frames = [tk.Frame(self.button_frame) for i in range(self.x)]
        self.buttons = [[tk.Button(self.frames[j], bg='white', width=int(30/10), height=int(30/10/2), command=partial(self.loc, x=i, y=j))
                         for j in range(self.x)] for i in range(self.y)]
        self.buttons[self.start[0]][self.start[1]].config(bg='red')
        self.buttons[self.goal[0]][self.goal[1]].config(bg='blue')
        for q in self.buttons:
            for w in q:
                w.pack(side='left')
        for z in self.frames:
            z.pack(fill='x')
        self.button_frame.pack(side='left')
        self.menu = tk.Frame(self.window)
        self.start_button = tk.Button(
            self.menu, text='Start Point', bg='red', width=10, height=5, command=partial(self.mode, mode='start'))
        self.obstacle_button = tk.Button(
            self.menu, text='Obstacle', bg='black', fg='white', width=10, height=5, command=partial(self.mode, mode='obstacle'))
        self.end_button = tk.Button(
            self.menu, text='End Point', bg='blue', width=10, height=5, command=partial(self.mode, mode='end'))
        self.default_button = tk.Button(
            self.menu, text='Reset', width=10, height=5, command=partial(self.mode, mode='reset'))
        self.random_button = tk.Button(
            self.menu, text='Random', width=10, height=5, command=partial(self.mode, mode='random'))
        self.search_button = tk.Button(
            self.menu, text='Search', bg='green', width=10, height=5, command=partial(self.mode))
        self.start_button.pack()
        self.obstacle_button.pack()
        self.end_button.pack()
        self.default_button.pack()
        self.random_button.pack()
        self.search_button.pack()
        self.menu.pack(fill='x', expand=True)

    def mode(self, mode=None):
        self.mode = mode
        if mode is None:
            route = self.astar(tuple(self.start), tuple(self.goal))
            self.route(route)
            for x in self.buttons:
                for y in x:
                    y.config(state='disabled')
        if mode == 'reset':
            self.grid = np.zeros((self.y, self.x), dtype=np.int32)
            result = np.where(self.grid == 0)
            all_result = list(zip(result[0], result[1]))
            for (x, y) in all_result:
                self.buttons[x][y].config(bg='white', state='normal')
            self.buttons[self.start[0]][self.start[1]].config(
                bg='red', state='normal')
            self.buttons[self.goal[0]][self.goal[1]].config(
                bg='blue', state='normal')
        if mode == 'random':
            limit = round((self.grid.shape[0]*self.grid.shape[1])/4)
            for x in range(limit):
                self.random(np.random.randint(0, 33), np.random.randint(0, 28))
            mask = np.random.randint(10, size=2)

    def random(self, x, y):
        if self.goal != [x, y] and self.start != [x, y]:
            self.grid[x][y] = 1
            self.buttons[x][y].config(bg='black')

    def route(self, route):
        if route:
            for x in route[1:-1]:
                self.turn_to_green(x[0], x[1])
                self.buttons[x[0]][x[1]].pack()

    def loc(self, x=None, y=None):
        if self.mode == 'start':
            if [x, y] == self.start or [x, y] == self.goal:
                pass
            else:
                self.buttons[self.start[0]][self.start[1]].config(bg='white')
                # print(x)
                self.turn_to_red(x, y)
                self.start[0] = x
                self.start[1] = y
        elif self.mode == 'obstacle':
            if [x, y] == self.start or [x, y] == self.goal:
                pass
            elif self.grid[x][y] == 1:
                self.grid[x][y] = 0
                self.buttons[x][y].config(bg='white')
            else:
                self.grid[x][y] = 1
                self.buttons[x][y].config(bg='black')
        elif self.mode == 'end':
            if [x, y] == self.start or [x, y] == self.goal:
                pass
            else:
                self.buttons[self.goal[0]][self.goal[1]].config(bg='white')
                # print(x)
                self.turn_to_blue(x, y)
                self.goal[0] = x
                self.goal[1] = y

        # print(self.start, self.goal)

    def turn_to_green(self, x, y):
        self.buttons[x][y].config(bg='green')

    def turn_to_yellow(self, x, y):
        self.buttons[x][y].config(bg='yellow')

    def turn_to_red(self, x, y):
        self.buttons[x][y].config(bg='red')

    def turn_to_blue(self, x, y):
        self.buttons[x][y].config(bg='blue')

    def heuristic(self, start, goal):
        return np.sqrt((goal[0] - start[0]) ** 2 + (goal[1] - start[1]) ** 2)

    def astar(self, start, goal):
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0),
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]

        close_set = set()
        came_from = {}

        gscore = {start: 0}
        fscore = {start: self.heuristic(start, goal)}

        oheap = []
        heapq.heappush(oheap, (fscore[start], start))

        while oheap:
            current = heapq.heappop(oheap)[1]
            if current == goal:
                self.turn_to_blue(goal[0], goal[1])
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                data = data + [start]
                return data[::-1]

            close_set.add(current)

            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                tentative_g_score = gscore[current] + \
                    self.heuristic(current, neighbor)

                if 0 <= neighbor[0] < self.grid.shape[0]:
                    if 0 <= neighbor[1] < self.grid.shape[1]:
                        if self.grid[neighbor[0]][neighbor[1]] == 1:
                            continue
                    else:
                        continue
                else:
                    continue

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue

                if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                    # self.turn_to_red(neighbor[0], neighbor[1])
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + \
                        self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
                self.turn_to_yellow(neighbor[0], neighbor[1])
                self.window.update()
        return False


def main():
    root = Field()
    root.window.mainloop()


if __name__ == "__main__":
    main()
