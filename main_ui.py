# Copyright Ashish Paudel, 2019. All rights reserved.
import matplotlib
import gc
import json

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.lines as lines
import matplotlib.pyplot as plt
import tkinter as tk
import os
import sys
from tkinter import ttk, filedialog
import copy
from main_clean import main as core_main


#constants
PAGE_TITLE = "LP Recognition"
FIG_SIZE = (12, 12)
DPI = 200
LOAD_BTN_TEXT = "Load Images"
PREV_BTN_TEXT = "Previous"
NEXT_BTN_TEXT = "Next"
RECT_FACE_COLOR = (1, 1, 1, 0)
RECT_EDGE_COLOR = "r"
LINE_WIDTH = 1
POINT_SIZE = 2
KEYCODE_ESCAPE = "escape"
BTNPRESS_EVENT_NAME = "button_press_event"
KEYPRESS_EVENT_NAME = "key_press_event"
MOUSEMOVE_EVENT_NAME = "motion_notify_event"
LARGE_FONT = ("Verdana", 12)
IMAGE_DIR = "raju"
GLOBAL_COUNTER = 0
matplotlib.use("TkAgg")
matplotlib.interactive(True)

class MainUI(tk.Tk):
    """
    The main application class.
    """

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, PAGE_TITLE)

        self.img = None
        self.f = Figure(figsize=FIG_SIZE, dpi=DPI)
        self.a = self.f.add_subplot(111)
        self.a.autoscale(tight=True)
        topFrame = tk.Frame(self)
        topFrame.pack()
        self.bind("<Return>", lambda e: self.onNext())
        # button1 = ttk.Button(topFrame, text=LOAD_BTN_TEXT,
        #                      command=lambda: self.onLoadImages())
        # button2 = ttk.Button(topFrame, text=PREV_BTN_TEXT,
        #                      command=lambda: self.onPrevious())
        # button3 = ttk.Button(topFrame, text=NEXT_BTN_TEXT,
        #                      command=lambda: self.onNext())
        # self.textEntry = tk.Entry(self)

        # button2.pack(side=tk.LEFT)
        # button1.pack(side=tk.LEFT)
        # button3.pack(side=tk.LEFT)
        self.agg = FigureCanvasTkAgg(self.f, self)
        self.agg.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.agg, self)
        toolbar.update()
        self.agg._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)





    def _refreshImage(self):
        self.a.imshow(self.img)
        self.agg.draw()





app = MainUI()
app.mainloop()


