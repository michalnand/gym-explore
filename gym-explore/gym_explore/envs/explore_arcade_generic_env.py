import gym
from gym import spaces
import numpy

from PIL import Image
import cv2
import time

class ExploreArcadeGenericEnv(gym.Env):
    def __init__(self, size = 4, room_size = 12, base_size = 8):
        gym.Env.__init__(self) 

        self._rnda      = 3141
        self._rndb      = 2718

        self.size       = size
        self.room_size  = room_size
        self.base_size  = base_size

        self.channels   = 3
        self.height     = self.room_size*self.base_size
        self.width      = self.room_size*self.base_size

        self.player_x_init = self.room_size//2
        self.player_y_init = self.room_size//2

        self.steps_max  = int(self.size*self.size*self.room_size*4)

        self.action_space       = spaces.Discrete(5)
        self.observation_space  = spaces.Box(low=-1.0, high=1.0,  shape=(self.channels, self.height, self.width), dtype=numpy.float32)

        k = 0.5
        self.colors = []
        self.colors.append([k*0.0, k*0.0, k*0.0])
        self.colors.append([k*0.0, k*0.0, k*0.5])
        self.colors.append([k*0.0, k*0.0, k*1.0])

        self.colors.append([k*0.0, k*0.5, k*0.0])
        self.colors.append([k*0.0, k*0.5, k*0.5])
        self.colors.append([k*0.0, k*0.5, k*1.0])

        self.colors.append([k*0.0, k*1.0, k*0.0])
        self.colors.append([k*0.0, k*1.0, k*0.5])
        self.colors.append([k*0.0, k*1.0, k*1.0])

        self.colors.append([k*0.5, k*0.0, k*0.0])
        self.colors.append([k*0.5, k*0.0, k*0.5])
        self.colors.append([k*0.5, k*0.0, k*1.0])

        self.colors.append([k*0.5, k*0.5, k*0.0])
        self.colors.append([k*0.5, k*0.5, k*0.5])
        self.colors.append([k*0.5, k*0.5, k*1.0])

        self.colors.append([k*0.5, k*1.0, k*0.0])
        self.colors.append([k*0.5, k*1.0, k*0.5])
        self.colors.append([k*0.5, k*1.0, k*1.0])

        self.colors.append([k*1.0, k*0.0, k*0.0])
        self.colors.append([k*1.0, k*0.0, k*0.5])
        self.colors.append([k*1.0, k*0.0, k*1.0])

        self.colors.append([k*1.0, k*0.5, k*0.0])
        self.colors.append([k*1.0, k*0.5, k*0.5])
        self.colors.append([k*1.0, k*0.5, k*1.0])

        self.colors.append([k*1.0, k*1.0, k*0.0])
        self.colors.append([k*1.0, k*1.0, k*0.5])

        self.colors.append([1.0, 1.0, 1.0])

        self.colors = numpy.array(self.colors)

        self.map = numpy.zeros((self.size, self.size, 3, self.room_size, self.room_size))
        
        self.backgrounds = numpy.zeros((self.size, self.size, 3))

        for ry in range(self.size):
            for rx in range(self.size):

                #fill background 
                color_idx = self._random_int()%(len(self.colors)-1)
                r = self.colors[color_idx][0]
                g = self.colors[color_idx][1]
                b = self.colors[color_idx][2]

                self.map[ry][rx][0] = r*numpy.ones((self.room_size, self.room_size))
                self.map[ry][rx][1] = g*numpy.ones((self.room_size, self.room_size))
                self.map[ry][rx][2] = b*numpy.ones((self.room_size, self.room_size))

                self.backgrounds[ry][rx][0] = r
                self.backgrounds[ry][rx][1] = g
                self.backgrounds[ry][rx][2] = b

                count = (10*self.room_size*self.room_size)//100

                for i in range(count):
                    color_idx   = self._random_int()%(len(self.colors)-1)
                    color       = self.colors[color_idx]
                    y           = self._random_int()%self.room_size
                    x           = self._random_int()%self.room_size

                    self.map[ry][rx][0][y][x] = color[0]
                    self.map[ry][rx][1][y][x] = color[1]
                    self.map[ry][rx][2][y][x] = color[2]
        
        self.map[0][0][0][self.player_y_init][self.player_x_init] = self.backgrounds[0][0][0]
        self.map[0][0][1][self.player_y_init][self.player_x_init] = self.backgrounds[0][0][1]
        self.map[0][0][2][self.player_y_init][self.player_x_init] = self.backgrounds[0][0][2]
              
    def reset(self):
        self._rnda  = 3141
        self._rndb  = 2718

        self.steps  = 0
        
        self.player_x = self.room_size//2
        self.player_y = self.room_size//2

        self.visited = numpy.zeros((self.size, self.size), dtype=bool)

        return self._update_observation()

    def step(self, action):

        self.steps+= 1

        reward = 0.0
        done   = False

        if action == 0:
            player_x = self.player_x + 1 
            player_y = self.player_y + 0
        elif action == 1:
            player_x = self.player_x - 1 
            player_y = self.player_y + 0
        elif action == 2:
            player_x = self.player_x + 0 
            player_y = self.player_y + 1
        elif action == 3:
            player_x = self.player_x + 0 
            player_y = self.player_y - 1
        else:
            player_x = self.player_x + 0 
            player_y = self.player_y + 0
        
        if player_x >= self.size*self.room_size-1:
            player_x = self.size*self.room_size-1

        if player_x < 0:
            player_x = 0

        if player_y >= self.size*self.room_size-1:
            player_y = self.size*self.room_size-1

        if player_y < 0:
            player_y = 0

        p_room_y = player_y//self.room_size
        p_room_x = player_x//self.room_size

        p_y = player_y%self.room_size
        p_x = player_x%self.room_size

        color_room  = self.backgrounds[p_room_y][p_room_x]
        color_field = [self.map[p_room_y][p_room_x][0][p_y][p_x], self.map[p_room_y][p_room_x][1][p_y][p_x], self.map[p_room_y][p_room_x][2][p_y][p_x]]

        dist = ((color_room - color_field)**2.0).sum()

        if dist < 0.01:
            self.player_y = player_y
            self.player_x = player_x

        self.visited[p_room_y][p_room_x] = True

        if self.steps >= self.steps_max:
            done      = True
            reward    = -1.0
        elif numpy.all(self.visited == True):
            done      = True
            reward    = 1.0

        return self._update_observation(), reward, done, None

    def render(self, full_view = True):

        if full_view:
            height = self.height + 2
            width  = self.width  + 2

            image = Image.new('RGB', (self.size*height, self.size*width))
            
            for ry in range(self.size):
                for rx in range(self.size):
                    im = self._get_room_image(ry, rx)
                    image.paste(im, (rx*width, ry*height))
        else:
            obs = self._update_observation()
            obs = numpy.moveaxis(obs, 0, -1)

            image = Image.fromarray(numpy.uint8(obs*255))

        cv2.imshow('env',numpy.array(image))
        cv2.waitKey(1)


    def _update_observation(self):

        p_room_y = self.player_y//self.room_size
        p_room_x = self.player_x//self.room_size

        im = self._get_room_image(p_room_y, p_room_x)

        frame   = numpy.array(im)/255.0
        frame   = numpy.moveaxis(frame, -1, 0)

        return frame

    def _get_room_image(self, room_y, room_x):
        p_room_y = self.player_y//self.room_size
        p_room_x = self.player_x//self.room_size

        p_y = self.player_y%self.room_size
        p_x = self.player_x%self.room_size

        tmp = self.map[room_y][room_x].copy()

        #put player if correct room
        if room_y == p_room_y and room_x == p_room_x:
            player_color    = len(self.colors)-1

            tmp[0][p_y][p_x] = self.colors[player_color][0]
            tmp[1][p_y][p_x] = self.colors[player_color][1]
            tmp[2][p_y][p_x] = self.colors[player_color][2]

        tmp = numpy.moveaxis(tmp, 0, -1)
        im  = Image.fromarray(numpy.uint8(tmp*255))
        im  = im.resize((self.height, self.width), Image.NEAREST)

        return im

    def _random_int(self):
        self._rnda = (1103515245*self._rnda + 12345)%(2**31)

        self._rndb ^= self._rndb >> 7
        self._rndb ^= self._rndb << 9
        self._rndb ^= self._rndb >> 13 

        return self._rnda + self._rndb


if __name__ == "__main__":
    env     = EnvRooms(size=4)
    state   = env.reset()

    fps = 0
    k = 0.02
    score = 0.0
    action = 1
    while True:
        if numpy.random.randint(10) == 0:
            action = numpy.random.randint(5)

        time_start = time.time()
        state, reward, done, _ = env.step(action)
        time_stop  = time.time()

        fps = (1.0-k)*fps + k*1.0/(time_stop - time_start)

        score+= reward

        if done:
            state = env.reset()
            print("score = ", score)
            print("fps   = ", fps)
            print("\n")

        env.render()
        time.sleep(0.01)
    



    