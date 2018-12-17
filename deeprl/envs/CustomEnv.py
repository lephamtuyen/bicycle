class CustomEnv(object):

    def __init__(self):
        self.goal = 0

    def reset(self):
        raise NotImplementedError("Not implemented")

    def step(self, action):
        raise NotImplementedError("Not implemented")
