import pygame
import time

from data_acquisition.config.config import *
from data_acquisition.modules.utils.stimulus.blankboard.Box import Box

from data_acquisition.modules.utils.Logger import Logger, app_logger
# logger = Logger(__name__)
logger = app_logger  # Log all in app.log


class BlankboardStimulus:
    def __init__(self, frequencies: dict, preparation_duration: int, stimulation_duration: int,
                 rest_duration: int, is_full_screen: bool = False):
        """

        :param frequencies: Dictionary of the boxes frequencies in this order (Top, Right, Down, Left)

        :param preparation_duration: The duration of the Epoc

        :param stimulation_duration: The duration of the Epoc

        :param rest_duration: The duration of the break

        :param is_full_screen: Indicator to run stimulus in a full-screen mode

        """
        self._done = False
        self._frequencies = frequencies
        self._preparation_duration = preparation_duration
        self._stimulation_duration = stimulation_duration
        self._rest_duration = rest_duration
        self._is_full_screen = is_full_screen

        self._screen_width = 1024
        self._screen_height = 780

        # Create the screen
        self._screen = pygame.display.set_mode((0, 0),
                                               pygame.FULLSCREEN) if self._is_full_screen else pygame.display.set_mode(
            [self._screen_width, self._screen_height])

        self._boxes = {}
        self._create_boxes()

    def _create_boxes(self):
        """
        Creates a list of boxes based on the position and frequency data stored in the _frequencies dictionary.
        :return:
        """
        screen_width = self._screen.get_width()
        screen_height = self._screen.get_height()

        for position, frequency in self._frequencies.items():
            box = Box(position, frequency, screen_width, screen_height)
            self._boxes[position] = box

    def run(self):
        pygame.init()

        for position in self._frequencies:
            logged = True
            logger.info(f"Start PREPARATION, {position} is now PURPLE")
            self._boxes[position].toggle_color()

            # Set the initial time
            start_time = time.time()

            # Main loop
            while not self._done:

                # Get the current time
                current_time = time.time()

                # Calculate the time elapsed since the last frame
                delta_time = current_time - start_time

                # Clear the screen
                self._screen.fill(BLACK)

                # # Displaying box info on the window for testing purpose.
                # self._display_info()

                # Toggle the box color after the preparation time.
                if time.time() - start_time >= self._preparation_duration and self._boxes[position].get_color() == PURPLE:
                    logger.info(f"End PREPARATION, {position} is now BLUE")
                    logger.info("Start STIMULATION")
                    self._boxes[position].toggle_color()

                # Displaying boxes until the stimulation time is ended.
                if time.time() - start_time < self._preparation_duration + self._stimulation_duration:

                    # Draw the box if the elapsed time is less than half the period
                    for box in self._boxes.values():
                        curr_frequency = box.get_frequency()
                        curr_color = box.get_color()

                        if delta_time % (1 / curr_frequency) < 1 / (2 * curr_frequency):
                            pygame.draw.rect(self._screen, curr_color, box.rect(), border_radius=BORDER_RADIUS)
                else:
                    if logged:
                        logger.info("End STIMULATION")
                        logger.info("Start REST")
                        logged = False

                # Stop the stimulus GUI after the session duration.
                if time.time() - start_time >= self._preparation_duration + self._stimulation_duration + self._rest_duration:
                    logger.info("End REST")
                    logger.info(f"End {position} session.")
                    break

                # Update the screen
                pygame.display.flip()

                for event in pygame.event.get():
                    if (event.type == pygame.KEYUP) or (event.type == pygame.KEYDOWN):
                        if event.key == pygame.K_ESCAPE:
                            self._done = True
                    if event.type == pygame.QUIT:
                        self._done = True
        pygame.quit()

    def close_stimulation(self):
        self._done = True

    def _display_info(self):
        for box in self._boxes.values():
            font = pygame.font.SysFont('Aerial', 30)
            text = font.render(f"{box.get_direction()} : {box.get_frequency()} HZ", False, WHITE)
            self._screen.blit(text, (box.get_left(), box.get_top()))


if __name__ == '__main__':
    BlankboardStimulus(FREQUENCIES_DICT, 1, 1, 1).run()
