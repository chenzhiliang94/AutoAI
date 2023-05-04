import numpy as np

class conditional1DNormal():
    bins = []
    mu = [0]
    std_dev = [1]
    def __init__(self, bins=[0.3], mu=[1,2],std_dev=[0.1,0.15]):
        assert len(bins) + 1 == len(mu)
        assert len(mu) == len(std_dev)
        self.bins = bins
        self.mu = mu
        self.std_dev = std_dev

    def func_ground_truth(self, x):
        for idx, interval in enumerate(self.bins):
            if x <= interval:
                return np.random.normal(loc=self.mu[idx], scale=self.std_dev[idx])

        return np.random.normal(loc=self.mu[-1], scale=self.std_dev[-1])

    def generate_ground_truth(self, input):
        f = np.vectorize(self.func_ground_truth)
        output = f(np.array(input))
        return output

    def generate_pertubed_component(self, input):
        # pertubed by a scaling factor
        return self.generate_ground_truth(input) * 0.9

    def generate_both(self, input):
        # returns both pertubed and non pertubed
        correct_output = self.generate_ground_truth(input)
        pertubed_output = self.generate_pertubed_component(input)
        return correct_output, pertubed_output #output + np.random.normal(0,0.1)