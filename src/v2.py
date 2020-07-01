import tkinter as tk
import numpy as np
import heapq
import time
from functools import partial


def mouse_event(event, mode=None):
    """
    turn path to obstacle, goal or start by click.
    return: None
    """
    global scale, start, goal, canvas, obstacle_variable
    if event.x > 1000 or event.y > 700:
        pass
    else:
        if mode == 'obstacle':
            if (event.x//scale[1], event.y//scale[1]) != start or (event.x//scale[1], event.y//scale[1]) != start:
                if grid[(event.x//scale[1])][(event.y//scale[1])] == 0:
                    canvas.create_rectangle(
                        ((event.x//scale[1])*scale[1],
                         (event.y//scale[1])*scale[1]),
                        ((event.x//scale[1])*scale[1]+scale[1],
                         (event.y//scale[1])*scale[1]+scale[1]),
                        outline='white',
                        fill='black')
                    grid[(event.x//scale[1])][(event.y//scale[1])] = 1
                    obstacle_variable.set(
                        "Obstacle : {} block".format((grid == 1).sum()))
                else:
                    canvas.create_rectangle(
                        ((event.x//scale[1])*scale[1],
                         (event.y//scale[1])*scale[1]),
                        ((event.x//scale[1])*scale[1]+scale[1],
                         (event.y//scale[1])*scale[1]+scale[1]),
                        outline='white',
                        fill=root.cget('bg'))
                    grid[(event.x//scale[1])][(event.y//scale[1])] = 0
                    obstacle_variable.set(
                        "Obstacle : {} block".format((grid == 1).sum()))

        elif mode == 'start':
            canvas.create_rectangle(
                ((start[0]*scale[1]),
                 (start[1]*scale[1])),
                ((start[0]*scale[1]+scale[1]),
                 (start[1]*scale[1]+scale[1])),
                outline='white',
                fill=root.cget('bg'))
            canvas.create_rectangle(
                ((event.x//scale[1])*scale[1],
                 (event.y//scale[1])*scale[1]),
                ((event.x//scale[1])*scale[1]+scale[1],
                 (event.y//scale[1])*scale[1]+scale[1]),
                outline='white',
                fill='red')
            start = ((event.x//scale[1]), (event.y//scale[1]))
        elif mode == 'goal':
            canvas.create_rectangle(
                ((goal[0]*scale[1]),
                 (goal[1]*scale[1])),
                ((goal[0]*scale[1]+scale[1]),
                 (goal[1]*scale[1]+scale[1])),
                outline='white',
                fill=root.cget('bg'))
            canvas.create_rectangle(
                ((event.x//scale[1])*scale[1],
                 (event.y//scale[1])*scale[1]),
                ((event.x//scale[1])*scale[1]+scale[1],
                 (event.y//scale[1])*scale[1]+scale[1]),
                outline='white',
                fill='green')
            goal = ((event.x//scale[1]), (event.y//scale[1]))


def motion(event):
    """
    turn path to obstacle by mouse motion.
    return: None
    """
    if event.x > 1000 or event.y > 700:
        pass
    else:
        if (event.x//scale[1], event.y//scale[1]) != start or (event.x//scale[1], event.y//scale[1]) != start:
            canvas.create_rectangle(
                ((event.x//scale[1])*scale[1],
                    (event.y//scale[1])*scale[1]),
                ((event.x//scale[1])*scale[1]+scale[1],
                    (event.y//scale[1])*scale[1]+scale[1]),
                outline='white',
                fill='black')
            grid[(event.x//scale[1])][(event.y//scale[1])] = 1
            obstacle_variable.set(
                "Obstacle : {} block".format((grid == 1).sum()))


def route(route):
    """
    create path line from start to route.
    return: None
    """
    global goal, scale
    if route:
        canvas.create_line((route[0][0][0]*scale[1])+(scale[1]/2), (route[0][0][1]*scale[1])+(scale[1]/2), (route[0][1][0]*scale[1]),
                           (route[0][1][1]*scale[1]), fill='blue', width=2.5)
        canvas.update()
        for [x, y] in route[1:]:
            canvas.create_oval(
                (x[0]*scale[1], x[1]*scale[1]), (x[0]*scale[1]+scale[1], x[1]*scale[1]+scale[1]), fill='cyan', outline='white')
            canvas.update()
        i = 0
        for [x, y] in route[1:]:
            canvas.create_line(x[0]*scale[1], x[1]*scale[1], y[0]*scale[1],
                               y[1]*scale[1], fill='blue', width=2.5)
            canvas.update()
            comment.insert(tk.END, "Step {}: ({}, {})\n".format(i, x[0], x[1]))
            comment.see(tk.END)
            i += 1
        comment.insert(
            tk.END, "Step {}: ({}, {})\n".format(i, goal[0], goal[1]))
        comment.see(tk.END)
        canvas.create_line((route[-1][0][0]*scale[1]), (route[-1][0][1]*scale[1]), (route[-1][1][0]*scale[1])+(scale[1]/2),
                           (route[-1][1][1]*scale[1])+(scale[1]/2), fill='blue', width=2.5)
        canvas.update()


def heuristic(start, goal):
    """
    heuristic scoring.
    return: score (float)
    """
    return np.sqrt((goal[0] - start[0]) ** 2 + (goal[1] - start[1]) ** 2)


def astar():
    """
    A Star Algorithm.
    return: list of route from start to goal (list)
    """
    global grid, start, goal
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0),
                 (1, 1), (1, -1), (-1, 1), (-1, -1)]

    close_set = set()
    came_from = {}

    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}

    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

    itera = 0
    while oheap:
        itera += 1
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append([came_from[current], current])
                current = came_from[current]
            data = data
            canvas.create_rectangle(
                goal[0]*scale[1], goal[1]*scale[1], goal[0]*scale[1]+scale[1], goal[1]*scale[1]+scale[1], outline='white', fill='green')
            return data[::-1]

        close_set.add(current)

        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)

            if 0 <= neighbor[0] < grid.shape[0]:
                if 0 <= neighbor[1] < grid.shape[1]:
                    if grid[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    continue
            else:
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + \
                    heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
            if grid[neighbor[0]][neighbor[1]] != 1:
                canvas.create_rectangle(
                    neighbor[0]*scale[1], neighbor[1]*scale[1], neighbor[0]*scale[1]+scale[1], neighbor[1]*scale[1]+scale[1], outline='white', fill='yellow')
            comment.insert(tk.END, "{} : Neighbors = {}, Score : {:.1f} at ({}, {})\n".format(
                itera, len(oheap), fscore[neighbor], neighbor[0], neighbor[1]))
            comment.see(tk.END)
            root.update()
    return False


def starts():
    """
    start function to start finding goal.
    return: None
    """
    global step_result
    step_result.set("Finding ...")
    routes = astar()
    if routes:
        step_result.set("Result {} step".format(len(routes)-1))
        comment.insert(tk.END, "Finish with {} step. from ({}. {}) to ({}, {}).".format(
            len(routes), start[0], start[1], goal[0], goal[1]))
        route(routes)
    else:
        step_result.set("Path Not Found")
        comment.insert(tk.END, "Path Not Found")


def random(size):
    """
    randomize obstacle of field. 1/4 of field
    return: None
    """
    comment.insert(tk.END, "Generating random ...")
    limit = round((grid.shape[0]*grid.shape[1])/size)
    step_result.set("Generating")
    for x in range(limit):
        step_result.set("Generating {:.2f}%".format((x/limit)*100))
        randomize(np.random.randint(
            0, grid.shape[0]), np.random.randint(0, grid.shape[1]))
        obstacle_variable.set(
            "Obstacle : {} block".format((grid == 1).sum()))
        canvas.update()
        comment.insert(tk.END, ".")
        comment.see(tk.END)
    comment.insert(tk.END, "\n")
    comment.insert(tk.END, "Obstacle {} block\n".format((grid == 1).sum()))
    step_result.set("Done")
    comment.insert(tk.END, "Done\n")
    comment.see(tk.END)


def randomize(x, y):
    """
    check if random path is start or goal and change color of path.
    return: None
    """
    if (x, y) != start and (x, y) != goal:
        grid[x][y] = 1
        canvas.create_rectangle(
            (x*scale[1], y*scale[1]), (x*scale[1]+scale[1], y*scale[1]+scale[1]), outline='white', fill='black')


def create_grid(event=None):
    """
    create grid line.
    return: None
    """
    w = canvas.winfo_width()
    h = canvas.winfo_height()
    canvas.delete('grid_line')

    for i in range(0, w, scale[1]):
        canvas.create_line([(i, 0), (i, h)], tag='grid_line',
                           fill='white', width=0.01)

    for i in range(0, h, scale[1]):
        canvas.create_line([(0, i), (w, i)], tag='grid_line',
                           fill='white', width=0.01)


def set_scale(size):
    """
    change grid variable and canvas grid size based on selected option.
    return: None
    """
    global scale, goal, grid, start_variable, goal_variable
    canvas.delete('all')
    scale = true_scale(size)
    grid = np.zeros((rescale_grid(scale[0])), dtype=np.int32)
    create_grid()
    goal = (X*scale[0]-1, Y*scale[0]-1)
    canvas.create_rectangle(
        (start[0]*scale[1], start[1]*scale[1]), (start[0]*scale[1]+scale[1], start[1]*scale[1]+scale[1]), fill='red', outline='white')
    canvas.create_rectangle(
        (goal[0]*scale[1], goal[1]*scale[1]), (goal[0]*scale[1]+scale[1], goal[1]*scale[1]+scale[1]), fill='green', outline='white')
    start_variable.set("Start Point : ({}, {})".format(start[0], start[1]))
    goal_variable.set("Goal Point : ({}, {})".format(goal[0], goal[1]))


def set_block(types):
    """
    set binding canvas as start, goal or obstacle types.
    return: None
    """
    global start, goal, is_normal
    if is_normal == True:
        is_normal = False
        canvas_scale_option.config(state="normal")
        start_button.config(state="normal")
        reset_button.config(state="normal")
        random_button_1.config(state="normal")
        random_button_2.config(state="normal")
        random_button_3.config(state="normal")
        if types == 'start':
            start_point_button.config(text="Set")
            end_point_button.config(state='normal')
            obstacle_button.config(state='normal')
            canvas.unbind("<Button-1>")
        elif types == 'goal':
            end_point_button.config(text="Set")
            start_point_button.config(state='normal')
            obstacle_button.config(state='normal')
            canvas.unbind("<Button-1>")
        elif types == 'obstacle':
            obstacle_button.config(text="Set")
            start_point_button.config(state='normal')
            end_point_button.config(state='normal')
            canvas.unbind("<Button-1>")
            canvas.unbind("<B1-Motion>")
    elif is_normal == False:
        is_normal = True
        canvas_scale_option.config(state="disabled")
        start_button.config(state="disabled")
        reset_button.config(state="disabled")
        random_button_1.config(state="disabled")
        random_button_2.config(state="disabled")
        random_button_3.config(state="disabled")
        if types == 'start':
            start_point_button.config(text="Done")
            end_point_button.config(state='disabled')
            obstacle_button.config(state='disabled')
            canvas.bind('<Button-1>', partial(mouse_event, mode='start'))
        elif types == 'goal':
            end_point_button.config(text="Done")
            start_point_button.config(state='disabled')
            obstacle_button.config(state='disabled')
            canvas.bind('<Button-1>', partial(mouse_event, mode='goal'))
        elif types == 'obstacle':
            obstacle_button.config(text="Done")
            start_point_button.config(state='disabled')
            end_point_button.config(state='disabled')
            canvas.bind('<B1-Motion>', motion)
            canvas.bind('<Button-1>', partial(mouse_event, mode='obstacle'))


def reset():
    """
    reset the grid variable and canvas grid obstacle.
    return: None
    """
    global grid
    grid = np.zeros((rescale_grid(scale[0])), dtype=np.int32)
    canvas.delete('all')
    create_grid()
    canvas.create_rectangle(
        (start[0]*scale[1], start[1]*scale[1]), (start[0]*scale[1]+scale[1], start[1]*scale[1]+scale[1]), fill='red', outline='yellow')
    canvas.create_rectangle(
        (goal[0]*scale[1], goal[1]*scale[1]), (goal[0]*scale[1]+scale[1], goal[1]*scale[1]+scale[1]), fill='green', outline='yellow')


def toggle_verbose():
    """
    toggler function to show or hide logging textarea.
    return: None
    """
    if show_verbose.get():
        comment.grid(row=1)
        comment.see(tk.END)
    else:
        comment.grid_forget()


def rescale_grid(scale):
    """
    change grid scale to selected option.
    return: coordinate based on canvas pixel (tuple)
    """
    return (X*scale, Y*scale)


def true_scale(scale):
    """
    change scaling grid to canvas coordinate.
    return: scale and distance pixel of grid (tuple)
    """
    return (scale, SCALE_OPTION[::-1][SCALE_OPTION.index(scale)])


# Set Constant Variables
(X, Y) = (10, 7)
SCALE_OPTION = (1, 2, 5, 10, 20, 50, 100)

# Main window tkinter
root = tk.Tk()
root.geometry("{0}x{1}+0+0".format(
    root.winfo_screenwidth()-3, root.winfo_screenheight()-3))
root.resizable(0, 0)
root.title("Pathfinding: A Start Algorithm")

# Set Variables
scale = true_scale(1)
start = (0, 0)
goal = (X*scale[0]-1, Y*scale[0]-1)
grid = np.zeros((rescale_grid(scale[0])), dtype=np.int32)
scale_variable = tk.IntVar()
scale_variable.set(SCALE_OPTION[0])
show_verbose = tk.BooleanVar()
step_result = tk.StringVar()
step_result.set("Press start to search route")
start_variable = tk.StringVar()
goal_variable = tk.StringVar()
obstacle_variable = tk.StringVar()
is_normal = False
opening_quote = """
Pathfinding Algorithm using A*
Press Start to begin searcing path
Press Reset to remove all obstacle in field
Press Random to generate random obstacle in field
Press Set on Obstacle then click or drag on grid to create your own obstacle
Press Set on Start or Goal button to set custom start and goal point
!!WARNING!!
BIGGER SCALE CAUSING PERFORMANCE OF APPS AND TAKING MORE TIME TO SOLVE PATHS
"""

# Grid Canvas
canvas = tk.Canvas(root, width=1000, height=700,
                   highlightthickness=1, highlightbackground="black")
canvas.create_rectangle(
    (start[0]*scale[1], start[1]*scale[1]), (start[0]*scale[1]+scale[1], start[1]*scale[1]+scale[1]), fill='red', outline='white')
canvas.create_rectangle(
    (goal[0]*scale[1], goal[1]*scale[1]), (goal[0]*scale[1]+scale[1], goal[1]*scale[1]+scale[1]), fill='green', outline='white')
canvas.bind('<Configure>', create_grid)
canvas.pack(side='left', padx=(10, 0))

# Option and Logging Frame
options = tk.Frame(root)

author = tk.Label(root, text="github.com/sagungt")
author.pack(side='top', anchor='ne')

# Option Menu Frame
title = tk.Label(options, text="A Star Algorithm", font=("helvetica", 20))
title.pack(side="top", pady=(100, 0))
menu = tk.Frame(options, width=100, height=100)

# Canvas scale input option
canvas_scale = tk.Label(
    menu, text="Size (10x7) Scale").grid(row=0, column=0, sticky='')
canvas_scale_option = tk.OptionMenu(
    menu, scale_variable, *SCALE_OPTION[:-1], command=set_scale)
canvas_scale_option.grid(row=0, column=1)

# Start and Goal Button
start_variable.set("Start Point : ({}, {})".format(start[0], start[1]))
start_point = tk.Label(menu,
                       textvariable=start_variable).grid(row=1, column=0)
start_point_button = tk.Button(
    menu, text='Set', command=partial(set_block, 'start'))
start_point_button.grid(row=1, column=1)

goal_variable.set("Goal Point : ({}, {})".format(goal[0], goal[1]))
end_point = tk.Label(menu,
                     textvariable=goal_variable).grid(row=2, column=0)
end_point_button = tk.Button(
    menu, text='Set', command=partial(set_block, 'goal'))
end_point_button.grid(row=2, column=1)

# Obstacle
obstacle_variable.set(
    "Obstacle : {} block".format((grid == 1).sum()))
obstacle = tk.Label(menu, textvariable=obstacle_variable).grid(row=3, column=0)
obstacle_button = tk.Button(menu, text="Set", command=partial(
    set_block, 'obstacle'))
obstacle_button.grid(row=3, column=1)

# Controller buttons
random_label = tk.Label(menu, text="Random ")
random_label.grid(row=4, column=0)
random_frame = tk.Frame(menu)
random_button_1 = tk.Button(
    random_frame, text='1/4', command=partial(random, 4))
random_button_1.pack(side='left')
random_button_2 = tk.Button(
    random_frame, text='1/6', command=partial(random, 6))
random_button_2.pack(side='left')
random_button_3 = tk.Button(
    random_frame, text='1/8', command=partial(random, 8))
random_button_3.pack(side='left')
random_frame.grid(row=4, column=1, columnspan=2)

start_button = tk.Button(menu, text='Start',
                         command=starts)
start_button.grid(row=5, column=0)
reset_button = tk.Button(
    menu, text='Reset', command=reset)
reset_button.grid(row=5, column=1)

# Verbose Checkbutton
verbose = tk.Checkbutton(menu, text='Verbose', variable=show_verbose, command=toggle_verbose).grid(
    row=6, column=0, columnspan=2)

# Logging Frame
result = tk.Frame(options)

# Simple logging label
text = tk.Label(result, textvariable=step_result).grid(
    row=0, column=0)

# Detail logging textarea
comment = tk.Text(result, width=46, height=10,
                  font=('courier', 8))
comment.insert(tk.END, opening_quote)

# Packing Frames
menu.pack(side='top', pady=(50, 0))
result.pack(side='top', expand=True, padx=(1, 0))
options.pack(side='top', padx=(3, 0))

root.mainloop()
