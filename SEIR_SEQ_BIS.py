import argparse

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
        self.sigma = 0.5        # Incubation rate
        self.gamma = 0.15       # Recovery rate
        self.hp = 0.05          # Hospit rate
        self.hcr = 0.2          # Hospit recovery rate
        self.pc = 0.1           # Critical rate
        self.pd = 0.1           # Critical recovery rate
        self.pcr = 0.3          # Critical mortality
        # Testing protocol
        self.s = 0.77          # Sensitivity
        self.t = 0.7           # Testing rate in symptomatical

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

        # Importance given to each curve during the fitting process
        self.w_1 = 1            # on test rate
        self.w_2 = 1            # Weight of positive test
        self.w_3 = 1            # Weight of cumul hospit data
        self.w_4 = 1
        self.w_5 = 1
        self.w_6 = 1

        # Value to return if log(binom.pmf(k,n,p)) = - infinity
        self.overflow = - 600

        # Smoothing data or not
        self.smoothing = False

        # Binomial smoother: ex: if = 2: predicted value *= 2 and p /= 2 WARNING: only use integer
        self.binom_smoother = 4

        # Binomial smoother use for model scoring:
        self.b_s_score = 2

        # Optimizer step size
        self.opti_step = 0.01

        # Optimizer constraints
        self.beta_min = 0.1
        self.beta_max = 0.6
        self.sigma_min = 1/5
        self.sigma_max = 1
        self.gamma_min = 1/10
        self.gamma_max = 1/4
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

        # ========================================== #
        #        Printers
        # ========================================== #
        self.fit_step_1_details = False


    def get_parameters(self):

        prm = (self.beta, self.sigma, self.gamma, self.hp, self.hcr, self.pc, self.pd, self.pcr, self.s, self.t)
        return prm

    def get_hyper_parameters(self):

        hprm = (self.w_1, self.w_2, self.w_3, self.w_4, self.w_5, self.w_6, self.binom_smoother, self.opti_step, self.optimizer, self.smoothing)
        return hprm

    def get_initial_state(self, sensib=None, test_rate=None, sigma=None):
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
        if sigma is None:
            sig = self.sigma
        else:
            sig = sigma

        I_0 = np.round(self.dataset[0][1])
        H_0 = self.dataset[0][3]
        E_0 = np.around((self.dataset[1][1] - self.dataset[0][1]) / sig)
        D_0 = 0
        C_0 = 0
        S_0 = 1000000 - I_0 - H_0 - E_0
        R_0 = 0
        dE_to_I_0 = np.around(self.dataset[0][1] / (s * t))
        dI_to_H_0 = H_0
        dI_to_R_0 = 0
        init = (S_0, E_0, I_0, R_0, H_0, C_0, D_0, dE_to_I_0, dI_to_H_0, dI_to_R_0)
        return init

    def differential(self, state, time, beta, sigma, gamma, hp, hcr, pc, pd, pcr, s, t):
        """
        ODE who describe the evolution of the model with the time
        :param state: An initial state to use
        :param time: A time vector
        :return: the evolution of the number of person in each compartiment + cumulative testing rate
        + cumulative entry in hospital
        """
        S, E, I, R, H, C, D, E_to_I, I_to_H, I_to_R = state

        dS = -(beta * S * I) / (S + I + E + R + H + C + D)
        dE = ((beta * S * I) / (S + I + E + R + H + C + D)) - (sigma * E)
        dI = (sigma * E) - (gamma * I) - (hp * I)
        dH = (hp * I) - (hcr * H) - (pc * H)
        dC = (pc * H) - (pd * C) - (pcr * C)
        dD = (pd * C)
        dR = (gamma * I) + (hcr * H) + (pcr * C)

        dE_to_I = sigma * E
        dI_to_H = hp * I
        dI_to_R = gamma * I

        return dS, dE, dI, dR, dH, dC, dD, dE_to_I, dI_to_H, dI_to_R

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
            init = self.get_initial_state(sensib=prm[8], test_rate=prm[9], sigma=prm[1])

        # Make prediction:
        predict = odeint(func=self.differential,
                         y0=init,
                         t=time,
                         args=(tuple(prm)))
        return predict

    def fit_and_select(self, part=1, optimizer='LBFGSB'):

        s_range = np.linspace(0.7, 0.85, 10)
        t_range = np.linspace(0.5, 1, 20)
        idx = 0
        total_idx = 100
        if part == 1:
            for s in range(0, len(s_range)):
                for t in range(0, 10):
                    params = (s_range[s], t_range[t])
                    idx += 1
                    print('=====================GENERAL ITER: {} / {}==================='.format(idx, total_idx))
                    self.fit_and_select_obj(params, part, optimizer)

        if part == 2:
            for s in range(0, len(s_range)):
                for t in range(10, 20):
                    params = (s_range[s], t_range[t])
                    idx += 1
                    print('=====================GENERAL ITER: {} / {}==================='.format(idx, total_idx))
                    self.fit_and_select_obj(params, part, optimizer)



    def fit_and_select_obj(self, parameters, part, optimizer):

        binom_smoother_range = [2, 3, 4, 5]
        idx = 0

        for bs in binom_smoother_range:
            idx += 1
            print('sub-iter: {} / {}'.format(idx, len(binom_smoother_range)))
            print('PART {} - Tested sensitivity: {}, tested testing rate: {} with binom smoother {} and optimizer {}'.format(part, parameters[0], parameters[1], bs, optimizer))
            model = SEIR()
            model.smoothing = False
            model.binom_smoother = bs
            model.import_dataset()
            model.optimizer = optimizer

            model.s = parameters[0]
            model.t = parameters[1]

            error = model.fit(display=False, method='method_2')
            print('Model rapport: ')
            parameters = model.get_parameters()
            print(parameters)
            print('error of the model: {}'.format(error))

            file = open('fit_result_part{}_opti_{}.csv'.format(part, optimizer), 'a')
            text = []
            text.append(str(float(error)))
            for item in parameters:
                text.append(str(item))
            text.append(str(model.binom_smoother))
            text.append(str(model.smoothing))
            text.append(str(model.optimizer))
            text.append(str(model.opti_step))

            final_str = ';'.join(text)
            file.write(final_str)
            file.write('\n')

            file.close()








    def fit(self, display=False, method='method_1'):
        """
        Compute best epidemic parameters values according to model's hyperparameters and the dataset
        """
        if method == 'method_1':
            # ======================================================================== #
            # First Step:
            # Fit the following parameters:
            # Beta, sigma, gamma, hp, t, s.
            # ======================================================================== #

            # Init to zero some parameters:
            self.hcr = 0
            self.pc = 0
            self.pd = 0
            self.pcr = 0
            # Get initial value for parameters:
            init_prm = (self.beta, self.sigma, self.gamma, self.hp, self.s, self.t)
            # Bounds
            bds = [(self.beta_min, self.beta_max),
                   (self.sigma_min, self.sigma_max),
                   (self.gamma_min, self.gamma_max),
                   (self.hp_min, self.hp_max),
                   (self.s_min, self.s_max),
                   (self.t_min, self.t_max)]
            # Constraint on parameters:
            cons = ({'type': 'ineq', 'fun': lambda x: -x[0] + self.beta_max},
                    {'type': 'ineq', 'fun': lambda x: -x[1] + self.sigma_max},
                    {'type': 'ineq', 'fun': lambda x: -x[2] + self.gamma_max},
                    {'type': 'ineq', 'fun': lambda x: -x[3] + self.hp_max},
                    {'type': 'ineq', 'fun': lambda x: -x[4] + self.s_max},
                    {'type': 'ineq', 'fun': lambda x: -x[5] + self.t_max},
                    {'type': 'ineq', 'fun': lambda x: x[0] - self.beta_min},
                    {'type': 'ineq', 'fun': lambda x: x[1] - self.sigma_min},
                    {'type': 'ineq', 'fun': lambda x: x[2] - self.gamma_min},
                    {'type': 'ineq', 'fun': lambda x: x[3] - self.hp_min},
                    {'type': 'ineq', 'fun': lambda x: x[4] - self.s_min},
                    {'type': 'ineq', 'fun': lambda x: x[5] - self.t_min})

            # Optimizer
            res = None
            if self.optimizer == 'LBFGSB':
                res = minimize(self.objective, np.asarray(init_prm),
                               method='L-BFGS-B',
                               options={'eps': self.opti_step},
                               constraints=cons,
                               bounds=bds,
                               args=('step_1', False, display))
            else:
                if self.optimizer == 'COBYLA':
                    res = minimize(self.objective, np.asarray(init_prm),
                                   method='COBYLA',
                                   args=('step_1', False, display),
                                   constraints=cons)
                else:  # Auto
                    res = minimize(self.objective, np.asarray(init_prm),
                                   constraints=cons,
                                   options={'eps': self.opti_step},
                                   args=('step_1', False, display))

            if display:
                # Print optimizer result
                print(res)

            # Update model parameters:
            self.beta = res.x[0]
            self.sigma = res.x[1]
            self.gamma = res.x[2]
            self.hp = res.x[3]
            self.s = res.x[4]
            self.t = res.x[5]

        if method == 'method_2':

            # ======================================================================== #
            # Method_2: sans S et T
            # Fit the following parameters:
            # Beta, sigma, gamma, hp, t, s.
            # ======================================================================== #

            # Init to zero some parameters:
            self.hcr = 0
            self.pc = 0
            self.pd = 0
            self.pcr = 0
            # Get initial value for parameters:
            init_prm = (self.beta, self.sigma, self.gamma, self.hp)
            # Bounds
            bds = [(self.beta_min, self.beta_max),
                   (self.sigma_min, self.sigma_max),
                   (self.gamma_min, self.gamma_max),
                   (self.hp_min, self.hp_max)]
            # Constraint on parameters:
            cons = ({'type': 'ineq', 'fun': lambda x: -x[0] + self.beta_max},
                    {'type': 'ineq', 'fun': lambda x: -x[1] + self.sigma_max},
                    {'type': 'ineq', 'fun': lambda x: -x[2] + self.gamma_max},
                    {'type': 'ineq', 'fun': lambda x: -x[3] + self.hp_max},
                    {'type': 'ineq', 'fun': lambda x: x[0] - self.beta_min},
                    {'type': 'ineq', 'fun': lambda x: x[1] - self.sigma_min},
                    {'type': 'ineq', 'fun': lambda x: x[2] - self.gamma_min},
                    {'type': 'ineq', 'fun': lambda x: x[3] - self.hp_min})

            # Optimizer
            res = None
            if self.optimizer == 'LBFGSB':
                res = minimize(self.objective, np.asarray(init_prm),
                               method='L-BFGS-B',
                               options={'eps': self.opti_step},
                               constraints=cons,
                               bounds=bds,
                               args=('step_1', False, display))
            else:
                if self.optimizer == 'COBYLA':
                    res = minimize(self.objective, np.asarray(init_prm),
                                   method='COBYLA',
                                   args=('step_1', False, display),
                                   constraints=cons)
                else:  # Auto
                    res = minimize(self.objective, np.asarray(init_prm),
                                   constraints=cons,
                                   options={'eps': self.opti_step},
                                   args=('step_1', False, display))

            if display:
                # Print optimizer result
                print(res)

            # Update model parameters:
            self.beta = res.x[0]
            self.sigma = res.x[1]
            self.gamma = res.x[2]
            self.hp = res.x[3]

        return res.fun


    def fit_rate_objectif(self, parameters):

        model = SEIR()
        model.set_param()
        model.s = 1
        model.t = parameters[0]
        model.import_dataset()
        model.fit()
        score = model.score(output='sum_tot')
        print('score for t= {}: {} '.format(parameters[0], - score))
        return - score







    def objective(self, parameters, method, print_details=False, display=False):

        if method == 'step_1':
            # Get full parameters:
            if len(parameters) == 6:
                params = (parameters[0], parameters[1], parameters[2], parameters[3], 0, 0, 0, 0, parameters[4], parameters[5])
            else:
                params = (parameters[0], parameters[1], parameters[2], parameters[3], 0, 0, 0, 0, self.s, self.t)
            # Get initial state:
            init_state = self.get_initial_state(sensib=params[-2], test_rate=params[-1], sigma=params[1])
            # Make predictions:
            pred = self.predict(duration=self.dataset.shape[0],
                                parameters=params,
                                initial_state=init_state)
            # Un-cumul tests predictions:
            uncumul = []
            uncumul.append(pred[0][7])
            for i in range(1, pred.shape[0]):
                uncumul.append(pred[i][7] - pred[i-1][7])
            if display:
                print(params)

            # Compute the joint probability of observations
            prb = 0
            for i in range(0, pred.shape[0]):

                # ======================================= #
                # PART 1: Fit testing rate by comparing
                # Test predictions and the number of test
                # ======================================= #
                n = np.around(uncumul[i] * self.binom_smoother)
                k = np.around(self.dataset[i][2])
                p = params[-1] / self.binom_smoother
                if n < 0:
                    p_k1 = np.log(binom.pmf(k=n, n=n, p=p))
                    if p_k1 == - math.inf:
                        p_k1 = self.overflow
                    p_k1 -= np.exp(k / (n*p))
                else:
                    if k > n:
                        p_k1 = self.overflow - np.fabs(n-k) ** 2
                    else:
                        p_k1 = np.log(binom.pmf(k=k, n=n, p=p))
                        if p_k1 == - math.inf:
                            p_k1 = self.overflow - np.fabs(n-k)
                #print('pred={}, n={}, n*p= {} k={}, p={}, p_k1={}'.format(uncumul[i], n, n*p, k, p, p_k1))
                prb -= p_k1 * self.w_1



                # ======================================= #
                # PART 2: fit on positive tests
                # ======================================= #
                p = params[-2] * params[-1]/ self.binom_smoother
                n = np.around(uncumul[i] * self.binom_smoother)
                k = np.around(self.dataset[i][1])
                if n < 0:
                    p_k2 = self.overflow - np.fabs(n-k) ** 2
                else:
                    if k > n:
                        p_k2 = np.log(binom.pmf(k=n, n=n, p=p))
                        if p_k2 == - math.inf:
                            p_k2 = self.overflow
                        p_k2 -= np.exp(k / (n * p))
                    else:
                        p_k2 = np.log(binom.pmf(k=k, n=n, p=p))
                        if p_k2 == - math.inf:
                            p_k2 = self.overflow - np.fabs(n-k)

                prb -= p_k2 * self.w_2

                # ======================================= #
                # PART 3: fit on I to H
                # ======================================= #
                n = np.around(pred[i][8] * self.binom_smoother)
                k = np.around(self.dataset[i][4])
                p = 1 / self.binom_smoother
                if n < 0:
                    p_k3 = self.overflow - np.fabs(n-k) ** 2
                else:
                    if k > n:
                        p_k3 = np.log(binom.pmf(k=n, n=n, p=p))
                        if p_k3 == - math.inf:
                            p_k3 = self.overflow
                        p_k3 -= np.exp(k / (n * p))
                    else:
                        p_k3 = np.log(binom.pmf(k=k, n=n, p=p))
                        if p_k3 == - math.inf:
                            p_k3 = self.overflow - np.fabs(n-k)

                prb -= p_k3 * self.w_3

                if self.fit_step_1_details:
                    print('iter {}: p_k1= {}, p_k2= {}, p_k3= {}'.format(i, p_k1, p_k2, p_k3))
                    print('test observ vs predict: {} - {}'.format(self.dataset[i][1], np.around(uncumul[i] * params[-1] * params[-2])))


            if display:
                print('loss: {}'.format(prb))
            #print('loss: {}'.format(prb))

            return prb




    def import_dataset(self):

        url = "https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"
        # Import the dataframe:
        raw = pd.read_csv(url, sep=',', header=0)
        raw['num_positive'][0] = 1
        raw['num_tested'][0] = 1
        # Ad a new column at the end with cumulative positive cases at the right
        cumul_positive = np.copy(raw['num_positive'].to_numpy())
        for i in range(1, len(cumul_positive)):
            cumul_positive[i] += cumul_positive[i-1]
        raw.insert(7, 'cumul_positive', cumul_positive)
        if self.smoothing:
            self.dataframe = dataframe_smoothing(raw)
        else: self.dataframe = raw
        self.dataset = self.dataframe.to_numpy()

        self.I_0 = self.dataset[0][1] / (self.s * self.t)
        self.E_0 = self.I_0 * 2
        self.R_0 = 0
        self.S_0 = 1000000 - self.I_0 - self.E_0



    def set_param(self):
        """
        Set the actual best values of parameters:

        Note: header for the result file:
        error;beta;sigma;gamma;hp;hcr;pc;pd;pcr;sensib;test_rate;binom_smoother;smoothing;optimizer;opti_step;
        """

        # Epidemic parameters:
        self.beta = 0.453638
        self.sigma = 0.885727
        self.gamma = 0.208646
        self.hp = 0.0207093
        self.hcr = 0.0313489
        self.pc = 0.0776738
        self.pd = 0.0417785
        self.pcr = 0.244847

def validation_result_analysis():

    result = pd.read_csv('validation_result.csv', sep=';')

    result.sort_values(by=['error'], inplace=True, ignore_index=True, ascending=True)

    print(result)
    npr = result.to_numpy()

    for i in range(0, npr.shape[0]):

        # Create a model:
        model = SEIR()

        # Load parameters:
        model.beta = npr[i][1]
        model.sigma = npr[i][2]
        model.gamma = npr[i][3]
        model.hp = npr[i][4]
        model.hcr = 0
        model.pc = 0
        model.pd = 0
        model.pcr = 0
        model.s = npr[i][9]
        model.t = npr[i][10]

        model.smoothing = False
        # Import dataset:
        model.import_dataset()

        # Make predictions:
        predictions = model.predict(duration=model.dataset.shape[0])
        # Uncumul
        uncumul = []
        uncumul.append(predictions[0][7])
        for j in range(1, predictions.shape[0]):
            uncumul.append(predictions[j][7] - predictions[j - 1][7])

        # Plot:
        time = model.dataset[:, 0]
        # Adapt test + with sensit and testing rate
        for j in range(0, len(time)):
            uncumul[j] = uncumul[j] * model.s * model.t

        # Plot cumul positive
        plt.scatter(time, model.dataset[:, 1], c='blue', label='test+')
        plt.plot(time, uncumul, c='blue', label='test+')
        # Plot hospit
        plt.scatter(time, model.dataset[:, 4], c='red', label='hospit cumul pred')
        plt.plot(time, predictions[:, 8], c='red', label='pred hopit cumul')
        plt.legend()
        plt.title('index {}'.format(i))
        plt.show()

        print('---------------------------------------------------------')

        row = result.loc[i, :]

        print(row)

        print("<Press enter/return to continue>")
        input()



def first():

    # Create the model:
    model = SEIR()
    # Import the dataset
    model.import_dataset()
    # Fit the model:
    model.fit(display=True)

    # Make pedictions:
    predictions = model.predict(model.dataset.shape[0])
    time = model.dataset[:, 0]
    # Uncumul
    uncumul = []
    uncumul.append(predictions[0][7])
    for j in range(1, predictions.shape[0]):
        uncumul.append(predictions[j][7] - predictions[j - 1][7])
    # Adapt test + with sensit and testing rate
    for j in range(0, len(time)):
        uncumul[j] = uncumul[j] * model.s * model.t


    # Plot cumul positive

    plt.scatter(time, model.dataset[:, 1], c='blue', label='test+')
    plt.plot(time, uncumul, c='blue', label='test+')
    # Plot hospit
    plt.scatter(time, model.dataset[:, 4], c='red', label='hospit cumul pred')
    plt.plot(time, predictions[:, 8], c='red', label='pred hopit cumul')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='blabla')

    parser.add_argument('--part', default='0')
    parser.add_argument('--opti', default='LBFGSB')

    args = parser.parse_args()


    if args.part == '1':
        if args.opti == 'LBFGSB':
            mdl = SEIR()
            mdl.fit_and_select(part=1, optimizer='LBFGSB')
        if args.opti == 'COBYLA':
            mdl = SEIR()
            mdl.fit_and_select(part=1, optimizer='COBYLA')
    if args.part == '2':
        if args.opti == 'LBFGSB':
            mdl = SEIR()
            mdl.fit_and_select(part=2, optimizer='LBFGSB')
        if args.opti == 'COBYLA':
            mdl = SEIR()
            mdl.fit_and_select(part=2, optimizer='COBYLA')

    #first()

    validation_result_analysis()



