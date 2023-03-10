from pygame import Surface, Rect


class Checkerboard:
    def create(self, noOfHorizontalBoxes: int = 5, noOfVerticalBoxes: int = 5, checkerboardWidth: int = 150,
               checkerboardHeight: int = 150) -> Surface:
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
