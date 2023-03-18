import pygame

from data_acquisition.config.config import *


class Box:
    def __init__(self, position, frequency, screen_width, screen_height):
        self._frequency = frequency
        self._position = position
        self._color = BLUE
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._left = 0
        self._top = 0

    def get_position(self, position):
        """
        get the left and top positions for a rectangle based on the desired position on the screen.

        :param position: The desired position of the rectangle, which can be one of the following constants: LEFT, RIGHT, TOP, DOWN, or CENTER.

        :return: A tuple containing the left and top positions of the rectangle
        """
        if position == LEFT_POSITION:
            left = self._screen_width / 2 - 2.5 * BOX_WIDTH
            top = self._screen_height / 2 - BOX_HEIGHT / 2
        elif position == RIGHT_POSITION:
            left = self._screen_width / 2 + 1.5 * BOX_WIDTH
            top = self._screen_height / 2 - BOX_HEIGHT / 2
        elif position == TOP_POSITION:
            top = self._screen_height / 2 - 2.5 * BOX_HEIGHT
            left = self._screen_width / 2 - BOX_WIDTH / 2
        elif position == DOWN_POSITION:
            top = self._screen_height / 2 + 1.5 * BOX_HEIGHT
            left = self._screen_width / 2 - BOX_WIDTH / 2
        elif position == CENTER_POSITION:
            left = self._screen_width / 2 - BOX_WIDTH / 2
            top = self._screen_height / 2 - BOX_HEIGHT / 2
        else:
            raise ValueError("location %s unknown" % position)

        return left, top

    def get_frequency(self):
        return self._frequency

    def get_color(self):
        return self._color

    def rect(self):
        self._left, self._top = self.get_position(self._position)
        rect = pygame.Rect(self._left, self._top, BOX_WIDTH, BOX_HEIGHT)
        return rect

    def get_left(self):
        return self._left

    def get_top(self):
        return self._top

    def get_direction(self):
        return self._position

    def toggle_color(self):
        """
        Toggle the color of the box between blue and purple.
        """
        self._color = BLUE if self._color == PURPLE else PURPLE
