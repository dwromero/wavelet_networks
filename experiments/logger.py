import sys
import os

class Logger(object):
    def __init__(self, args):
        self.terminal = sys.stdout

        # Incremental name (Avoid erasing existing files)
        i = 1
        while os.path.exists(args.path[:-4] + "logfile%s.out" % i):
            i += 1

        self.path = args.path[:-4] + "logfile%s.out" % i
        self.log = open(self.path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

