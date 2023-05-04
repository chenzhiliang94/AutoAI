import numpy as np

class DifferentiablePolynomial():
    '''
    x^1.15-1.4x+e^-0.3x
    '''

    def func(self, x):
        '''

        :param x: scalar form
        :return: function
        '''
        return -(x)**2+1.4*x+np.exp(0.3 * x)

    def get_gradients_default(self, x):
        '''

        :param x: x input
        :return: gradient
        '''
        # this is just dy/dx with x given
        return -2 * (x) ** 1 + 1.4 - 0.3 * np.exp(-0.3 * x)

    def generate_ground_truth(self, input):
        '''

        :param input: vector of X
        :return: function vector
        '''
        f = np.vectorize(self.func)
        output = f(np.array(input))
        return output

    def generate_pertubed_component(self, input):
        '''

        :param input: vector of X
        :return: pertubed function vector (erronous)
        '''
        # pertubed by a scaling factor
        np.random.seed(1)
        return self.generate_ground_truth(input) * np.random.normal(1,0.05)

    def generate_both(self, input):
        # returns both pertubed and non pertubed
        correct_output = self.generate_ground_truth(input)
        pertubed_output = self.generate_pertubed_component(input)
        return correct_output, pertubed_output #output + np.random.normal(0,0.1)