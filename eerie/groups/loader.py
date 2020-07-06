import importlib


class Group:
    def __init__(self, group_name):
        super(Group, self).__init__()
        group_lib = importlib.import_module('eerie.groups.' + group_name)
        self.H = group_lib.H()
        self.Rd = group_lib.Rd()
        self.G = group_lib.G()

        self.h_grid_global = group_lib.h_grid_global
        self.h_grid_local = group_lib.h_grid_local
