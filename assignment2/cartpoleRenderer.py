import pygame
import numpy as np
import math


class CartPole2DEnvRenderer:
    def __init__(self, env):
        self.isopen = None
        pygame.init()
        self.env = env
        self.screen_width = 600
        self.screen_height = 400
        self.pole_width = 10
        self.pole_length = 100
        self.cart_width = 60
        self.cart_height = 30
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

    def render(self):
        if self.env.state is None:
            return

        # Check if the display surface is still initialized.
        if not pygame.display.get_init():
            return

        if self.screen is None or not self.isopen:
            # Attempt to re-initialize Pygame and the display surface if it was previously closed.
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.isopen = True

        self.screen.fill((255, 255, 255))

        x, _, y, _, theta_x, _, theta_y, _ = self.env.state

        # Convert state to screen position
        cart_x = int(x * self.screen_width / (2 * self.env.x_threshold) + self.screen_width / 2.0)
        cart_y = int(y * self.screen_height / (2 * self.env.x_threshold) + self.screen_height / 2.0)

        # Clear screen
        self.screen.fill((255, 255, 255))

        # Draw cart
        cart_rect = pygame.Rect(cart_x - self.cart_width / 2, cart_y - self.cart_height / 2,
                                self.cart_width, self.cart_height)
        pygame.draw.rect(self.screen, (0, 0, 255), cart_rect)

        # Draw pole
        pole_end_x = cart_x + self.pole_length * np.sin(theta_x)
        pole_end_y = cart_y - self.pole_length * np.cos(theta_y)  # In pygame, the y-axis is inverted
        pygame.draw.line(self.screen, (255, 0, 0), (cart_x, cart_y), (pole_end_x, pole_end_y), self.pole_width)

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()
