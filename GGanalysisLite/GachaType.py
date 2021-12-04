import numpy as np
from scipy import signal

class PityGacha():
    def __init__(self, pity_p=None):
        if pity_p is None:
            self.pity_p = self.generate_pity_p()
        else:
            self.pity_p = pity_p
        self.distribution = self.calc_distribution(self.pity_p)
    def generate_pity_p(self):
        pity_p = np.zeros(91, dtype=float)
        for i in range(1, 74):
            pity_p[i] = 0.006
        for i in range(74, 91):
            pity_p[i] = 0.06 + pity_p[i-1]
        pity_p[90] = 1
        return pity_p
    def calc_distribution(self, pity_p=[0, 1]):
        item_distribution = np.zeros(len(pity_p), dtype=float)
        temp_state = 1
        for i in range(1, len(pity_p)):
            item_distribution[i] = temp_state * pity_p[i]
            temp_state = temp_state * (1-pity_p[i])
        return item_distribution
    def calc_expectation(self, item_distribution=None):
        if item_distribution is None:
            item_distribution = self.distribution
        item_expectation = 0
        for i in range(1, len(item_distribution)):
            item_expectation += item_distribution[i] * i
        return item_expectation

class Pity5starCharacter(PityGacha):
    pass

class Up5starCharacter(Pity5starCharacter):
    def __init__(self, pity_p=None):
        super().__init__(pity_p)
        temp_distribution = signal.convolve(self.distribution, self.distribution)
        self.distribution = np.pad(self.distribution, (0, len(self.pity_p)-1), 'constant')
        self.distribution = (self.distribution + temp_distribution)/2

class Pity5starWeapon(PityGacha):
    def generate_pity_p(self):
        pity_p = np.zeros(81, dtype=float)
        for i in range(1, 63):
            pity_p[i] = 0.007
        for i in range(63, 74):
            pity_p[i] = pity_p[i-1] + 0.07
        for i in range(74, 80):
            pity_p[i] = pity_p[i-1] + 0.035
        pity_p[80] = 1
        return pity_p

if __name__ == '__main__':
    a = Pity5starWeapon()
    print(a.calc_expectation())