
class Flicky(object):
    def __init__(self, x, y, images, frames=10):
        self.IMAGES = images
        self.x = x
        self.y = y
        self.w = self.IMAGES[0].get_width()
        self.h = self.IMAGES[0].get_width()
        self.frames = frames
        self.clock = 0
        self.img_index = 0

    def process(self):
        self.clock = self.clock + 1
        if self.clock >= self.frames:
            self.clock = 0
            self.img_index = 0 if self.img_index == 1 else 1

    def draw(self, screen):
        screen.blit(self.IMAGES[self.img_index], (self.x, self.y))
