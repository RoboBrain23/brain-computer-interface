from data_acquisition.gui.widgets.stimulus.Checkerboard import Checkerboard
from data_acquisition.gui.widgets.stimulus.Flicky import Flicky

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

IMAGES = [
            Checkerboard.create(0),
            Checkerboard.create(1),
        ]
A = IMAGES[0].get_width()


class Stimulus:

    def __init__(self, screen):
        self.flickies = []
        self.screen = screen

        # Box position constants
        self.TOP = 0
        self.RIGHT = 1
        self.DOWN = 2
        self.LEFT = 3
        self.CENTER = 4

    def addFlicky(self, f):
        self.flickies.append(f)

    def add(self, position, frames):
        w, h = self.screen.get_size()
        if position == self.LEFT:
            x = 0
            y = h / 2 - A / 2
        elif position == self.RIGHT:
            x = w - A
            y = h / 2 - A / 2
        elif position == self.TOP:
            y = 0
            x = w / 2 - A / 2
        elif position == self.DOWN:
            y = h - A
            x = w / 2 - A / 2
        elif position == self.CENTER:
            x = w / 2 - A / 2
            y = h / 2 - A / 2
        else:
            raise ValueError("location %s unknown" % position)

        f = Flicky(x, y, IMAGES, frames)
        self.flickies.append(f)

    def process(self):
        for f in self.flickies:
            f.process()

    def draw(self):
        for f in self.flickies:
            f.draw(self.screen)
