import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import math
from scipy.stats import binom as binom

from smoothing import dataframe_smoothing

class SEIR():

    def __init__(self):

        # ========================================== #
        #           Model parameters
        # ========================================== #
        self.beta = 0.3         # Contamination rate
        self.sigma = 0.8        # Incubation rate
        self.gamma = 0.15       # Recovery rate
        self.hp = 0         # Hospit rate
        self.hcr = 0          # Hospit recovery rate
        self.pc = 0          # Critical rate
        self.pd = 0           # Critical recovery rate
        self.pcr = 0         # Critical mortality

        # Learning set
        self.dataframe = None
        self.dataset = None

        # Initial state
        self.I_0 = 2                        # Infected
        self.E_0 = 3                        # Exposed
        self.R_0 = 0                        # Recovered
        self.S_0 = 1000000 - self.I_0 - self.E_0      # Sensible
        self.H_0 = 0
        self.C_0 = 0
        self.D_0 = 0
        self.CT_0 = self.I_0                # Contamined

        # ========================================== #
        #        Hyperparameters dashboard:
        # ========================================== #

        # Testing protocol
        self.s = 0.75          # Sensitivity
        self.t = 0.6           # Testing rate in symptomatical

        # Importance given to each curve during the fitting process
        self.w_1 = 1          # Weight of cumulative positive data
        self.w_2 = 0.5         # Weight of hopit data
        self.w_3 = 0.5          # Weight of cumul hospit data
        self.w_4 = 1            # Weight àf critical data
        self.w_5 = 1            # Weight of fatalities data

        # Value to return if log(binom.pmf(k,n,p)) = - infinity
        self.overflow = - 1000

        # Smoothing data or not
        self.smoothing = True

        # Binomial smoother: ex: if = 2: predicted value *= 2 and p /= 2 WARNING: only use integer
        self.binom_smoother = 4

        # Binomial smoother use for model scoring:
        self.b_s_score = 2

        # Optimizer step size
        self.opti_step = 0.01

        # Optimizer constraints
        self.beta_min = 0.1
        self.beta_max = 0.9
        self.sigma_min = 1/5
        self.sigma_max = 1
        self.gamma_min = 0.01
        self.gamma_max = 1/2
        self.hp_min = 0.01
        self.hp_max = 0.5
        self.hcr_min = 0.01
        self.hcr_max = 0.4
        self.pc_min = 0.01
        self.pc_max = 0.4
        self.pd_min = 0.01
        self.pd_max = 0.5
        self.pcr_min = 0.01
        self.pcr_max = 0.4
        self.s_min = 0.7
        self.s_max = 0.85
        self.t_min = 0.5
        self.t_max = 1

        # Optimizer choise: COBYLA LBFGSB ou AUTO
        self.optimizer = 'LBFGSB'

        # Fit type:
        self.fit_type = 'type_1'

    def get_parameters(self):

        prm = (self.beta, self.sigma, self.gamma, self.hp, self.hcr, self.pc, self.pd, self.pcr, self.s, self.t)
        return prm

    def get_hyper_parameters(self):

        hprm = (self.s, self.t, self.w_1, self.w_2, self.w_3, self.w_4, self.w_5, self.binom_smoother, self.opti_step, self.optimizer, self.smoothing)
        return hprm

    def get_initial_state(self, sensib=None, test_rate=None):
        """
        Generate an initial state for the model from the dataset
        according to the sensitivity and the testing rate to
        estimate the true value of the initial state
        :param sensib: Sensibility value to use. Use class value if None
        :param test_rate: Testing rate value to use. Use class value if None
        :return: An array
        """
        if sensib is None:
            s = self.s
        else:
            s = sensib
        if test_rate is None:
            t = self.t
        else:
            t = test_rate

        #I_0 = np.round(np.round(self.dataset[0][1] / (s * t)))
        I_0 = np.round(4 / (s * t))
        H_0 = self.dataset[0][3]
        E_0 = 2 * I_0
        D_0 = 0
        C_0 = 0
        S_0 = 1000000 - I_0 - H_0 - E_0
        R_0 = 0
        CT_0 = I_0
        CH_0 = H_0
        init = (S_0, E_0, I_0, R_0, H_0, C_0, D_0, CT_0, CH_0)
        return init

    def differential(self, state, time, beta, sigma, gamma, hp, hcr, pc, pd, pcr, s, t):
        """
        ODE who describe the evolution of the model with the time
        :param state: An initial state to use
        :param time: A time vector
        :return: the evolution of the number of person in each compartiment + cumulative testing rate
        + cumulative entry in hospital
        """
        S, E, I, R, H, C, D, CT, CH = state

        dS = -(beta * S * I) / (S + I + E + R + H + C + D)
        dE = ((beta * S * I) / (S + I + E + R + H + C + D)) - (sigma * E)
        dI = (sigma * E) - (gamma * I) - (hp * I)
        dH = (hp * I) - (hcr * H) - (pc * H)
        dC = (pc * H) - (pd * C) - (pcr * C)
        dD = (pd * C)
        dR = (gamma * I) + (hcr * H) + (pcr * C)

        dCT = sigma * E
        dCH = hp * I

        return dS, dE, dI, dR, dH, dC, dD, dCT, dCH

    def predict(self, duration, initial_state=None, parameters=None):
        """
        Predict the evolution of the epidemic during the selected duration from a given initial state
        and given parameters
        :param duration: Use positive integer value
        :param initial_state: Default = use self.get_initial_state()
        :param parameters: Default = use self.get_parameters()
        :return: a numpy array of 8 columns and t rows
        """
        # Time vector:
        time = np.arange(duration)
        # Parameters to use
        prm = parameters
        if prm is None:
            prm = self.get_parameters()
        # Initial state to use:
        init = initial_state
        if init is None:
            init = self.get_initial_state()

        # Make prediction:
        predict = odeint(func=self.differential,
                         y0=init,
                         t=time,
                         args=(tuple(prm)))
        return predict

    def fit_and_select(self, optimizer='LBFGSB', silent=False):
        """
        Model selection:
        Optimize sensitivity and testing rate value on basis model
        :return:
        """
        # set start values of parameters:
        init_prm = (self.s, self.t)
        # Bounds:
        bds = [(self.s_min, self.s_max), (self.t_min, self.t_max)]
        # Constraint on parameters:
        cons = ({'type': 'ineq', 'fun': lambda x: -x[0] + self.s_max},
                {'type': 'ineq', 'fun': lambda x: -x[1] + self.t_max},
                {'type': 'ineq', 'fun': lambda x: x[0] - self.s_max},
                {'type': 'ineq', 'fun': lambda x: x[1] - self.t_max})

        # Optimizer
        res = None
        if optimizer == 'LBFGSB':
            res = minimize(self.select_objective, np.asarray(init_prm),
                           method='L-BFGS-B',
                           options={'eps': self.opti_step},
                           constraints=cons,
                           bounds=bds)
        else:
            if optimizer == 'COBYLA':
                res = minimize(self.select_objective, np.asarray(init_prm),
                               method='COBYLA',
                               constraints=cons)
            else:  # Auto
                res = minimize(select_objective, np.asarray(init_prm),
                               constraints=cons,
                               options={'eps': self.opti_step})
        # Print optimizer result
        if not silent:
            print("--------------------------------------------------")
            print('Model selection result ')
            print(res)

        self.s = res.x[0]
        self.t = res.x[1]

        # Train a model with theses values:
        self.fit()



    def select_objective(self, parameters):

        # Create a model:
        model = SEIR()
        # Set parameters:
        model.s = parameters[0]
        model.t = parameters[1]
        # Inport the dataset:
        model.import_dataset()
        print("======================================")
        print('Start to fit a model with: ')
        print('Sensitivity = {}, testing rate = {}'.format(parameters[0], parameters[1]))

        model.fit(silent=False)

        # Score the model
        score = model.score(output='sum_tot', method='method_1')
        print('Score of the model: {}'.format(score))
        print("======================================")
        return score

    def fit(self, silent=False):
        """
        Compute best epidemic parameters values according to model's hyperparameters and the dataset
        """
        if self.fit_type == 'type_1':
            # Initial values of parameters:
            init_prm = (self.beta, self.sigma, self.gamma)
            # Time vector:
            time = self.dataset[:, 0]
            # Bounds
            bds = [(self.beta_min, self.beta_max), (self.sigma_min, self.sigma_max), (self.gamma_min, self.gamma_max)]
            # Constraint on parameters:
            cons = ({'type': 'ineq', 'fun': lambda x: -x[0] + self.beta_max},
                    {'type': 'ineq', 'fun': lambda x: -x[1] + self.sigma_max},
                    {'type': 'ineq', 'fun': lambda x: -x[2] + self.gamma_max},
                    {'type': 'ineq', 'fun': lambda x: x[0] - self.beta_min},
                    {'type': 'ineq', 'fun': lambda x: x[1] - self.sigma_min},
                    {'type': 'ineq', 'fun': lambda x: x[2] - self.gamma_min})

            # Optimizer
            res = None
            if self.optimizer == 'LBFGSB':
                res = minimize(self.objective, np.asarray(init_prm),
                               method='L-BFGS-B',
                               options={'eps': self.opti_step},
                               constraints=cons,
                               bounds=bds,
                               args=('method_1'))
            else:
                if self.optimizer == 'COBYLA':
                    res = minimize(self.objective, np.asarray(init_prm),
                                   method='COBYLA',
                                   args=('method_1'),
                                   constraints=cons)
                else:   # Auto
                    res = minimize(self.objective, np.asarray(init_prm),
                                   constraints=cons,
                                   options={'eps': self.opti_step},
                                   args=('method_1'))


            # Print optimizer result
            if not silent:
                print(res)
            # Update model parameters:
            self.beta = res.x[0]
            self.sigma = res.x[1]
            self.gamma = res.x[2]



    def objective(self, parameters, method, print_details=False, silent=False):
        """
        The objective function to minimize during the fitting process.
        These function compute the probability of each observed values accroding to predictions
        take the logarighm value and make the sum.
        """

        if method == 'method_1':
            # Make predictions:
            prms = tuple(parameters)
            params = (prms[0], prms[1], prms[2], 0, 0, 0, 0, 0, self.s, self.t)
            init_state = self.get_initial_state()
            if not silent:
                print(params)
            pred = self.predict(duration=self.dataset.shape[0],
                                parameters=params,
                                initial_state=init_state)
            # Uncumul positive test:
            uncumul = []
            uncumul.append(pred[0][7])
            for i in range(1, pred.shape[0]):
                uncumul.append(pred[i][7] - pred[i-1][7])
            # Compare with dataset:
            prb = 0
            for i in range(0, pred.shape[0]):
                p_k1 = p_k2 = self.overflow
                # ======================================= #
                # PART 1: Fit on positive test
                # ======================================= #
                pa = self.s * self.t
                n = np.around(uncumul[i])
                k = self.dataset[i][1]
                p = 1 / self.binom_smoother * pa
                if k < 0 and n < 0:
                    k *= -1
                    n *= -1
                if k > n:
                    tmp = n
                    n = k
                    k = tmp
                if k < 0:
                    n += - k + 1
                    k = 1
                n *= self.binom_smoother
                prob = binom.pmf(k=k, n=n, p=p)
                if prob > 0:
                    p_k1 = np.log(binom.pmf(k=k, n=n, p=p))
                else:
                    p_k1 = self.overflow
                prb -= p_k1 * self.w_1

                if print_details:
                    print('iter {}: {}'.format(i, p_k1))
                    print('test+ cumul: {} - {}'.format(np.around(pred[i][7] * params[8] * params[9]), self.dataset[i][7]))
            if not silent:
                print(prb)
            return prb

        if method == 'method_2':
            # Make predictions:
            prms = tuple(parameters)
            params = (prms[0], prms[1], prms[2], 0, 0, 0, 0, 0, self.s, self.t)
            init_state = self.get_initial_state()
            if not silent:
                print(params)
            pred = self.predict(duration=self.dataset.shape[0],
                                parameters=params,
                                initial_state=init_state)
            # Uncumul positives
            uncum = []
            uncum.append(pred[0][7])
            for i in range(1, pred.shape[0]):
                uncum.append(pred[i][7] - pred[i-1][7])
            # Compare with dataset with SSE
            prb = 0
            for i in range(0, pred.shape[0]):
                # ======================================= #
                # PART 1: Fit on positive test
                # ======================================= #
                pa = 1
                n = uncum[i] * pa
                k = self.dataset[i][1]

                #value = np.log((n _ k) ** 2)
                prb += np.fabs(n - k)


            if not silent:
                print(prb)
            return prb


    def score(self, output='raw', method='method_1'):

        if method == 'method_1':
            # Get parameters
            params = tuple(self.get_parameters())
            # Get Initial state
            init_state = self.get_initial_state()
            pred = self.predict(duration=self.dataset.shape[0],
                                parameters=params,
                                initial_state=init_state)
            # Store raw result:
            raw = np.zeros((pred.shape[0], 4))
            # Compare with dataset:
            for i in range(0, pred.shape[0]):
                p_k1 = self.overflow

                # ======================================= #
                # PART 1: Compare cumul positives
                # ======================================= #
                n = np.around(pred[i][7] * self.s * self.t)
                k = self.dataset[i][7]
                p = 1 / self.b_s_score
                if k < 0 and n < 0:
                    k *= -1
                    n *= -1
                if k > n:
                    tmp = n
                    n = k
                    k = tmp
                if k < 0:
                    n += - k + 1
                    k = 1
                n *= self.b_s_score
                prob = binom.pmf(k=k, n=n, p=p)
                if prob > 0:
                    p_k1 = np.log(binom.pmf(k=k, n=n, p=p))
                else:
                    p_k1 = self.overflow
                raw[i][0] = p_k1

            if output == 'raw':
                return raw

            if output == 'sum_tot':
                return np.sum(raw)

    def opti_rates(self):
        """
        Optimize the values of sensitivity and testing rate after pré-fit
        """
        pass



    def import_dataset(self):

        url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
        # Import the dataframe:
        raw = pd.read_csv(url, sep=',', header=0)
        raw['num_positive'][0] = 1
        # Ad a new column at the end with cumulative positive cases at the right
        cumul_positive = np.copy(raw['num_positive'].to_numpy())
        for i in range(1, len(cumul_positive)):
            cumul_positive[i] += cumul_positive[i-1]
        raw.insert(7, 'cumul_positive', cumul_positive)
        if self.smoothing:
            self.dataframe = dataframe_smoothing(raw)
        else:
            self.dataframe = raw
        self.dataset = self.dataframe.to_numpy()

        self.I_0 = self.dataset[0][1] / (self.s * self.t)
        self.E_0 = self.I_0 * 5
        self.R_0 = 0
        self.S_0 = 1000000 - self.I_0 - self.E_0



    def set_param(self):
        """
        Set the actual best values of parameters:

        Note: header for the result file:
        sum_tot;beta;sigma;gamma;hp;hcr;pc;pd;pcr;sensib;test_rate;w1;w2;w3;w4;w5;binom_smoother;opti_step;optimizer;smoothing;mean_tot_bis;sum_tot;std_tot;mean_test;sum_test;std_test;mean_hospit;sum_hospit;std_hospit;mean_critical;sum_critical;std_critical;mean_fata;sum_fata;std_fata
        """

        # Epidemic parameters:

        self.beta = 0.545717
        self.sigma = 0.778919
        self.gamma = 0.215823
        self.hp = 0.196489
        self.hcr = 0.0514182
        self.pc = 0.075996
        self.pd = 0.0458608
        self.pcr = 0.293681
        self.s = 0.765
        self.t = 0.75

        # Hyper parameters:
        self.s = 0.765
        self.t = 0.75
        self.w_1 = 2
        self.w_2 = 1
        self.w_3 = 1
        self.w_4 = 2
        self.w_5 = 1
        self.binom_smoother = 3
        self.opti_step = 0.1
        self.optimizer = 'COBYLA'
        self.smoothing = False


def valid_result_analysis():

    # Import validation result:
    result = pd.read_csv('validation_result.csv', sep=';')

    result.sort_values(by=['sum_tot', 'std_tot'], inplace=True, ignore_index=True, ascending=False)
    print(result)

    # Numpy version:
    npr = result.to_numpy()

    # exec:
    for i in range(0, npr.shape[0]):

        # Create a model:
        model = SEIR()

        # Load parameters:
        model.beta = npr[i][1]
        model.sigma = npr[i][2]
        model.gamma = npr[i][3]
        model.hp = npr[i][4]
        model.hcr = npr[i][5]
        model.pc = npr[i][6]
        model.pd = npr[i][7]
        model.pcr = npr[i][8]
        model.s = npr[i][9]
        model.t = npr[i][10]

        model.optimizer = 'COBYLA'
        model.smoothing = False

        # Import dataset:
        model.import_dataset()

        # Make predictions:
        predictions = model.predict(duration=model.dataset.shape[0])


        # Plot:
        time = model.dataset[:, 0]
        # Adapt test + with sensit and testing rate
        positive_cumul = []
        for j in range(0, len(time)):
            positive_cumul.append(predictions[j][7] * model.s * model.t)

        # Plot cumul positive
        plt.scatter(time, model.dataset[:, 7], c='blue', label='cumul test+')
        plt.plot(time, positive_cumul, c='blue', label='cumul test+')
        # Plot hospit
        plt.scatter(time, model.dataset[:, 4], c='red', label='hospit cumul pred')
        plt.plot(time, predictions[:, 8], c='red', label='pred hopit cumul')
        plt.legend()
        plt.title('index {}'.format(i))
        plt.show()

        # Plot critical
        plt.scatter(time, model.dataset[:, 5], c='green', label='critical data')
        plt.plot(time, predictions[:, 5], c='green', label='critical pred')
        plt.scatter(time, model.dataset[:, 6], c='black', label='fatalities data')
        plt.plot(time, predictions[:, 6], c='black', label='fatalities pred')
        plt.legend()
        plt.title('index {}'.format(i))
        plt.show()

        print('---------------------------------------------------------')

        row = result.loc[i, :]

        print(row)

        print("<Press enter/return to continue>")
        input()








if __name__ == "__main__":

    # Create the model:
    model = SEIR()
    # Import dataset:
    model.import_dataset()

    # Fit:
    #model.fit_and_select()
    model.fit()


    params = model.get_parameters()




    # Make a prediction:
    prd = model.predict(model.dataset.shape[0], parameters=params)

    # Uncumul:
    uncumul = []
    uncumul.append(prd[0][7])
    for i in range(1, prd.shape[0]):
        uncumul.append(prd[i][7] - prd[i-1][7])

    for i in range(0, prd.shape[0]):
        uncumul[i] = uncumul[i] * model.s * model.t


    print('=== For positif: ')
    for i in range(0, 10):
        print('dataset: {}, predict = {}'.format(model.dataset[i, 1], uncumul[i]))
    print('=== For hospit: ')

    #print(prd[:, 1])

    # Plot
    plt.scatter(model.dataset[:, 0], model.dataset[:, 1], c='blue', label='testing data')
    plt.scatter(model.dataset[:, 0], model.dataset[:, 4], c='green', label='hospit_cum')
    plt.plot(model.dataset[:, 0], prd[:, 4], c='yellow', label='hospit cum pred')
    plt.plot(model.dataset[:, 0], uncumul, c='red', label='predictions')
    plt.legend()
    plt.show()


"""    plt.scatter(model.dataset[:, 0], model.dataset[:, 5], c='blue', label='critical data')
    plt.plot(model.dataset[:, 0], prd[:, 5], c='red', label='critical prediction')
    plt.scatter(model.dataset[:, 0], model.dataset[:, 6], c='yellow', label='dead data')
    plt.plot(model.dataset[:, 0], prd[:, 6], c='green', label='dead predict')
    plt.legend()
    plt.show()
"""
