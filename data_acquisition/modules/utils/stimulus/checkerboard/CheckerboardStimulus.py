import time

import pygame

from data_acquisition.modules.utils.stimulus.checkerboard.Checkerboard import Checkerboard
from data_acquisition.modules.utils.stimulus.checkerboard.Flicky import Flicky

from data_acquisition.modules.utils.Logger import Logger, app_logger
logger = Logger(__name__)   # Log in {current_filename}.log
# logger = app_logger   # Log all in app.log


_IMAGES = [
    Checkerboard.create(0),
    Checkerboard.create(1),
]
_A = _IMAGES[0].get_width()


class CheckerboardStimulus:

    def __init__(self, is_full_screen: bool, stimulus_screen_width: int, stimulus_screen_height: int, frequencies: dict, stimulation_duration: int,
                 rest_duration: int):
        """

        :param is_full_screen: Indicator to run stimulus in a full-screen mode

        :param stimulus_screen_width: The width of the stimulus screen

        :param stimulus_screen_height: The Height of the stimulus screen

        :param frequencies: Dictionary of the boxes frequencies in this order (Top, Right, Down, Left)

        :param stimulation_duration: The duration of the Epoc

        :param rest_duration: The duration of the break

        """
        self._flickies = []

        self._is_full_screen = is_full_screen
        self._stimulus_screen_width = stimulus_screen_width
        self._stimulus_screen_height = stimulus_screen_height
        self._display = [self._stimulus_screen_width, self._stimulus_screen_height]

        self._clock = None
        self._done = False
        self._screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN) if self._is_full_screen \
            else pygame.display.set_mode(self._display)

        # Box position constants
        self.TOP = 0
        self.RIGHT = 1
        self.DOWN = 2
        self.LEFT = 3
        self.CENTER = 4

        # Stimulus Configurations
        self._frequencies = frequencies

        self.TOP_FREQ = frequencies["top"]
        self.RIGHT_FREQ = frequencies["right"]
        self.DOWN_FREQ = frequencies["down"]
        self.LEFT_FREQ = frequencies["left"]

        self._stimulation_duration = stimulation_duration
        self._rest_duration = rest_duration * 1000

    def run(self, is_training: bool):
        """
        Initialize and display the SSVEP Stimulus GUI.

        :param is_training: bool value to give an indication to run training stimulus mode.
        """

        pygame.init()
        pygame.display.set_caption("SSVEP Stimulus")

        self._done = False
        self._clock = pygame.time.Clock()

        for direction, frame in self._frequencies.items():
            self._done = False

            if is_training:
                self.add(self.CENTER, frame)
            else:
                # Controlling mode is ON
                self.add(self.TOP, self.TOP_FREQ)
                self.add(self.RIGHT, self.RIGHT_FREQ)
                self.add(self.DOWN, self.DOWN_FREQ)
                self.add(self.LEFT, self.LEFT_FREQ)

            starting_time = time.time()
            logger.info(f"Current direction {direction} with the {60/frame} HZ")
            while not self._done:
                if is_training:
                    if (time.time() - starting_time) >= self._stimulation_duration:
                        logger.info(f"Start rest duration {self._rest_duration/1000} sec")
                        pygame.time.wait(self._rest_duration)
                        logger.info(f"End of rest duration")
                        break

                for event in pygame.event.get():
                    if (event.type == pygame.KEYUP) or (event.type == pygame.KEYDOWN):
                        if event.key == pygame.K_ESCAPE:
                            self._done = True
                    if event.type == pygame.QUIT:
                        self._done = True

                self._screen.fill((0, 0, 0))
                self._clock.tick(60)  # 16 ms between frames ~ 60FPS
                self.process()
                self.draw()
                pygame.display.flip()
            if not is_training or self._done:
                break
        pygame.quit()

    def addFlicky(self, f):
        self._flickies.append(f)

    def add(self, position, frames):
        w, h = self._screen.get_size()
        if position == self.LEFT:
            x = 0
            y = h / 2 - _A / 2
        elif position == self.RIGHT:
            x = w - _A
            y = h / 2 - _A / 2
        elif position == self.TOP:
            y = 0
            x = w / 2 - _A / 2
        elif position == self.DOWN:
            y = h - _A
            x = w / 2 - _A / 2
        elif position == self.CENTER:
            x = w / 2 - _A / 2
            y = h / 2 - _A / 2
        else:
            raise ValueError("location %s unknown" % position)

        f = Flicky(x, y, _IMAGES, frames)
        self._flickies.append(f)

    def process(self):
        for f in self._flickies:
            f.process()

    def draw(self):
        for f in self._flickies:
            f.draw(self._screen)
