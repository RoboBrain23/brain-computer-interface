import time

import pygame

from data_acquisition.gui.widgets.stimulus.Checkerboard import Checkerboard
from data_acquisition.gui.widgets.stimulus.Flicky import Flicky


IMAGES = [
    Checkerboard.create(0),
    Checkerboard.create(1),
]
A = IMAGES[0].get_width()


class Stimulus:

    def __init__(self, screen, isFullScreen, stimulusScreenWidth, stimulusScreenHeight, frequencies, epoc_duration, break_duration):

        self.flickies = []
        self.screen = screen

        self.isFullScreen = isFullScreen
        self.stimulusScreenWidth = stimulusScreenWidth
        self.stimulusScreenHeight = stimulusScreenHeight
        self.display = [self.stimulusScreenWidth, self.stimulusScreenHeight]

        self.clock = None
        self.done = False

        # Box position constants
        self.TOP = 0
        self.RIGHT = 1
        self.DOWN = 2
        self.LEFT = 3
        self.CENTER = 4

        # Stimulus Configurations
        self.frequencies = frequencies

        self.TOP_FREQ = frequencies[0]
        self.RIGHT_FREQ = frequencies[1]
        self.DOWN_FREQ = frequencies[2]
        self.LEFT_FREQ = frequencies[3]

        self.EPOC_DURATION = epoc_duration  # Seconds
        self.BREAK_DURATION = break_duration  # Milliseconds

    def run(self, isTraining: bool):
        """
        Initialize and display the SSVEP Stimulus GUI.

        :param isTraining: bool value to give an indication to run training stimulus mode.
        """

        pygame.init()
        pygame.display.set_caption("SSVEP Stimulus")
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN) if self.isFullScreen else pygame.display.set_mode(self.display)

        self.done = False
        self.clock = pygame.time.Clock()

        for i in range(len(self.frequencies)):
            self.done = False

            if isTraining:
                self.add(self.CENTER, self.frequencies[i])
            else:
                # Controlling mode is ON
                self.add(self.TOP, self.TOP_FREQ)
                self.add(self.RIGHT, self.RIGHT_FREQ)
                self.add(self.DOWN, self.DOWN_FREQ)
                self.add(self.LEFT, self.LEFT_FREQ)

            startingTime = time.time()
            while not self.done:
                if isTraining:
                    if (time.time() - startingTime) >= self.EPOC_DURATION:
                        if i < len(self.frequencies) - 1:
                            pygame.time.wait(self.BREAK_DURATION)
                        break

                for event in pygame.event.get():
                    if (event.type == pygame.KEYUP) or (event.type == pygame.KEYDOWN):
                        if event.key == pygame.K_ESCAPE:
                            self.done = True
                    if event.type == pygame.QUIT:
                        self.done = True

                self.screen.fill((0, 0, 0))
                self.clock.tick(60)  # 16 ms between frames ~ 60FPS
                self.process()
                self.draw()
                pygame.display.flip()

            if not isTraining or self.done:
                break
        pygame.quit()

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
