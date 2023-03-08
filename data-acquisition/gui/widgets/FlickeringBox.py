from pygame import Surface, Rect


class Checkerboard:
    def create(self, noOfHorizontalBoxes: int = 5, noOfVerticalBoxes: int = 5, checkerboardWidth: int = 100, checkerboardHeight: int = 100) -> Surface:
        """

        :param noOfHorizontalBoxes: Number of boxes in the vertical line or X-axis in the checkerboard.
        :param noOfVerticalBoxes: Number of boxes in the horizontal line or Y-axis in the checkerboard.
        :param checkerboardWidth: The width checkerboard.
        :param checkerboardHeight: The height of checkerboard.
        :return: pygame.Surface which a pygame object for representing images.
        """
        boxWidth = checkerboardWidth / noOfHorizontalBoxes
        boxHeight = checkerboardHeight / noOfVerticalBoxes

        white = (255, 255, 255)
        black = (0, 0, 0)

        surf = Surface((checkerboardWidth, checkerboardHeight))
        boxColor = black if self else white
        surf.fill(boxColor)

        for i in range(0, noOfHorizontalBoxes):
            for j in range(0, noOfVerticalBoxes):
                boxColor = white if boxColor == black else black
                rect = Rect(boxWidth * i, boxHeight * j, boxWidth, boxHeight)
                surf.fill(boxColor, rect)
        return surf


IMAGES = [
    Checkerboard.create(0),
    Checkerboard.create(1),
]
A = IMAGES[0].get_width()


class FlickeringManager:

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

        f = Flicky(x, y, frames)
        self.flickies.append(f)

    def process(self):
        for f in self.flickies:
            f.process()

    def draw(self):
        for f in self.flickies:
            f.draw(self.screen)


class Flicky(object):
    def __init__(self, x, y, frames=10, w=A, h=A):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.frames = frames
        self.clock = 0
        self.img_index = 0

    def process(self):
        self.clock = self.clock + 1
        if self.clock >= self.frames:
            self.clock = 0
            self.img_index = 0 if self.img_index == 1 else 1

    def draw(self, screen):
        screen.blit(IMAGES[self.img_index], (self.x, self.y))
