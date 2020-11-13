from typing import (
    Tuple,
)

import numpy
import torch
import random

from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
)

P = [0.082712, 0.124068, 0.151639, 0.172317, 0.188859, 0.202644, 0.214460, 0.224799, 0.233990, 0.242261, 0.249780,
     0.256673, 0.263035, 0.268943, 0.274457, 0.279627, 0.284492, 0.289087, 0.293441, 0.297576, 0.301515, 0.305274,
     0.308871, 0.312317, 0.315625, 0.318807, 0.321870, 0.324824, 0.327676, 0.330433, 0.333101, 0.335686, 0.338193,
     0.340625, 0.342989, 0.345286, 0.347522, 0.349698, 0.351819, 0.353887, 0.355904, 0.357873, 0.359797, 0.361677,
     0.363515, 0.365313, 0.367073, 0.368796, 0.370484, 0.372138, 0.373760, 0.375351, 0.376911, 0.378443, 0.379947,
     0.381424, 0.382875, 0.384301, 0.385703, 0.387081, 0.388437, 0.389771, 0.391084, 0.392377, 0.393649, 0.394902,
     0.396137, 0.397353, 0.398552, 0.399734, 0.400898, 0.402047, 0.403180, 0.404298, 0.405401, 0.406489, 0.407563,
     0.408624, 0.409671, 0.410705, 0.411726, 0.412734, 0.413731, 0.414716, 0.415689, 0.416651, 0.417601, 0.418541,
     0.419470, 0.420390, 0.421298, 0.422197, 0.423087, 0.423967, 0.424837, 0.425699, 0.426552, 0.427396, 0.428231,
     0.429058, 0.429877, 0.430688, 0.431491, 0.432286, 0.433074, 0.433855, 0.434628, 0.435393, 0.436152, 0.436904,
     0.437649, 0.438388, 0.439120, 0.439845, 0.440565, 0.441278, 0.441984, 0.442685, 0.443380, 0.444070, 0.444753,
     0.445431, 0.446104, 0.446771, 0.447432, 0.448089, 0.448740, 0.449386, 0.450028, 0.450664, 0.451295, 0.451922,
     0.452544, 0.453161, 0.453774, 0.454382, 0.454986, 0.455585, 0.456180, 0.456771, 0.457357, 0.457940, 0.458518,
     0.459093, 0.459663, 0.460230, 0.460792, 0.461351, 0.461906, 0.462458, 0.463005, 0.463550, 0.464090, 0.464627,
     0.465161, 0.465691, 0.466218, 0.466741, 0.467262, 0.467779, 0.468292, 0.468803, 0.469310, 0.469815, 0.470316,
     0.470814, 0.471309, 0.471802, 0.472291, 0.472778, 0.473261, 0.473742, 0.474220, 0.474696, 0.475168, 0.475638,
     0.476106, 0.476570, 0.477032, 0.477492, 0.477949, 0.478403, 0.478855, 0.479305, 0.479752, 0.480197, 0.480639,
     0.481079, 0.481517, 0.481952, 0.482385, 0.482816, 0.483244, 0.483671, 0.484095, 0.484517, 0.484937, 0.485354,
     0.485770, 0.486184]


class ReplayMemory(object):

    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
    ) -> None:
        self.__device = device
        self.__capacity = capacity
        self.__size = 0
        self.__pos = 0
        self.__peroid = capacity // 100

        self.__m_states = torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8)
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.__m_exp = numpy.ndarray((capacity,), dtype=float)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.__indices = torch.zeros((32, 1), dtype=torch.int)
        self.__a = numpy.ndarray((capacity,), dtype=int)

    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
            exp: float
    ) -> None:
        self.__m_states[self.__pos] = folded_state
        self.__m_actions[self.__pos, 0] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done
        self.__m_exp[self.__pos] = exp

        self.__pos = (self.__pos + 1) % self.__capacity
        self.__size = max(self.__size, self.__pos)

    def sample(self, batch_size: int):
        self.__indices = torch.randint(0, high=self.__size, size=(batch_size,))
        self.__a = numpy.argsort(-self.__m_exp)
        for j in range(batch_size):
            p = random.uniform(0, 1)
            for i in range(200):
                if (p <= P[i]):
                    self.__indices[j] = self.__a[i]
                    break
        b_state = self.__m_states[self.__indices, :4].to(self.__device).float()
        b_next = self.__m_states[self.__indices, 1:].to(self.__device).float()
        b_action = self.__m_actions[self.__indices].to(self.__device)
        b_reward = self.__m_rewards[self.__indices].to(self.__device).float()
        b_done = self.__m_dones[self.__indices].to(self.__device).float()
        return b_state, b_action, b_reward, b_next, b_done, self.__indices

    def recal_flag(self):
        if self.__pos % self.__peroid == 0:
            if self.__pos < self.__size:
                return self.__pos + 1, min(self.__pos + 1000, self.__capacity)
            return 0, min(self.__pos, 1000)
        return 0, 0

    # def select(self, i):
    #     return self.__m_states[i], self.__m_actions[i, 0].item(), self.__m_rewards[i, 0].item(), \
    #            self.__m_dones[i, 0].item()

    def update(self, i, exp):
        self.__m_exp[i] = exp.item()

    def __len__(self) -> int:
        return self.__size
