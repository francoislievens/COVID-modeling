import numpy as np
import matplotlib.pyplot as plt
import SEIR_2
import argparse


class model_selection():

    def __init__(self, smooth=False):

        # ========================================== #
        #           Hyper-Parameters values to test
        # ========================================== #

        self.w_1 = [1, 2]
        self.w_2 = [1, 2]
        self.w_3 = [1, 2]
        self.w_4 = [1, 2]
        self.w_5 = [1, 2]

        # Binomial smoother
        self.binom_smoother = [2, 3, 4, 5]

        # Optimizer to choose:
        self.optimizer = 'LBFGSB'

        # Step size (if LBFGSB)
        self.step_size = [0.1, 0.05, 0.01]

        self.model = SEIR_2.SEIR()

        self.smoothing = smooth
        if self.smoothing:
            self.model.smoothing = True
        else:
            self.model.smoothing = False
        self.model.import_dataset()
        self.model.fit_type = 'type_1'

    def perform(self, select_part=1, optimizer='LBFGSB'):

        iter = 0
        total_iter = len(self.w_1) ** 4 * len(self.binom_smoother)
        if optimizer=='LBFGSB':
            total_iter *= len(self.step_size)

        self.model.optimizer = optimizer
        if select_part == 1:
            self.model.w_1 = self.w_1[0]
        else:
            self.model.w_1 = self.w_1[1]

        for w_2 in self.w_2:
            self.model.w_2 = w_2
            for w_3 in self.w_3:
                self.model.w_3 = w_3
                for w_4 in self.w_4:
                    self.model.w_4 = w_4
                    for w_5 in self.w_5:
                        self.model.w_5 = w_5
                        for binom_smoother in self.binom_smoother:
                            self.model.binom_smoother = binom_smoother

                            if self.model.optimizer == 'LBFGSB':
                                for step_size in self.step_size:
                                    self.model.opti_step = step_size

                                    iter += 1
                                    print('iter {} / {}'.format(iter, total_iter))

                                    # Reinit the model
                                    self.model.beta = 0.3  # Contamination rate
                                    self.model.sigma = 0.8  # Incubation rate
                                    self.model.gamma = 0.15  # Recovery rate
                                    self.model.hp = 0.05  # Hospit rate
                                    self.model.hcr = 0.2  # Hospit recovery rate
                                    self.model.pc = 0.1  # Critical rate
                                    self.model.pd = 0.1  # Critical recovery rate
                                    self.model.pcr = 0.3  # Critical mortality
                                    self.model.s = 0.765  # Sensitivity
                                    self.model.t = 0.75  # Testing rate in symptomatical

                                    # Fit the model:
                                    self.model.fit()

                                    # Get SEIR parameters value:
                                    param_seir = self.model.get_parameters()

                                    # Get model's Hyper parameters
                                    h_param = self.model.get_hyper_parameters()

                                    # Get score:
                                    raw = self.model.score(output='raw')
                                    mean_test = str(np.mean(raw[:, 0]))
                                    sum_test = str(np.sum(raw[:, 0]))
                                    std_test = str(np.std(raw[:, 0]))
                                    mean_hospit = str(np.mean(raw[:, 1]))
                                    sum_hospit = str(np.sum(raw[:, 1]))
                                    std_hospit = str(np.std(raw[:, 1]))
                                    mean_critical = str(np.mean(raw[:, 2]))
                                    sum_critical = str(np.sum(raw[:, 2]))
                                    std_critical = str(np.std(raw[:, 2]))
                                    mean_fata = str(np.mean(raw[:, 3]))
                                    sum_fata = str(np.sum(raw[:, 3]))
                                    std_fata = str(np.std(raw[:, 3]))
                                    mean_tot = str(np.mean(raw))
                                    sum_tot = str(np.sum(raw))
                                    std_tot = str(np.std(raw))

                                    # Write in file:
                                    # Make a list of informations:
                                    str_lst = []
                                    str_lst.append(sum_tot)
                                    for item in param_seir:
                                        str_lst.append(item)
                                    for item in h_param:
                                        str_lst.append(item)
                                    str_lst.append(mean_tot)
                                    str_lst.append(sum_tot)
                                    str_lst.append(std_tot)
                                    str_lst.append(mean_test)
                                    str_lst.append(sum_test)
                                    str_lst.append(std_test)
                                    str_lst.append(mean_hospit)
                                    str_lst.append(sum_hospit)
                                    str_lst.append(std_hospit)
                                    str_lst.append(mean_critical)
                                    str_lst.append(sum_critical)
                                    str_lst.append(std_critical)
                                    str_lst.append(mean_fata)
                                    str_lst.append(sum_fata)
                                    str_lst.append(std_fata)

                                    convert_str_lst = []
                                    for i in range(0, len(str_lst)):
                                        convert_str_lst.append(str(str_lst[i]))

                                    print(convert_str_lst)

                                    final_str = ';'.join(convert_str_lst)

                                    file = open(
                                        'mod_select_result_part{}-{}-{}.csv'.format(select_part, self.smoothing, self.model.optimizer), "a")
                                    file.write(final_str)
                                    file.write('\n')

                                    file.close()

                            else:

                                iter += 1
                                print('iter {} / {}'.format(iter, total_iter))

                                # Reinit the model
                                self.model.beta = 0.3  # Contamination rate
                                self.model.sigma = 0.8  # Incubation rate
                                self.model.gamma = 0.15  # Recovery rate
                                self.model.hp = 0.05  # Hospit rate
                                self.model.hcr = 0.2  # Hospit recovery rate
                                self.model.pc = 0.1  # Critical rate
                                self.model.pd = 0.1  # Critical recovery rate
                                self.model.pcr = 0.3  # Critical mortality
                                self.model.s = 0.765  # Sensitivity
                                self.model.t = 0.75  # Testing rate in symptomatical

                                # Fit the model:
                                self.model.fit()

                                # Get SEIR parameters value:
                                param_seir = self.model.get_parameters()

                                # Get model's Hyper parameters
                                h_param = self.model.get_hyper_parameters()

                                # Get score:
                                raw = self.model.score(output='raw')
                                mean_test = str(np.mean(raw[:, 0]))
                                sum_test = str(np.sum(raw[:, 0]))
                                std_test = str(np.std(raw[:, 0]))
                                mean_hospit = str(np.mean(raw[:, 1]))
                                sum_hospit = str(np.sum(raw[:, 1]))
                                std_hospit = str(np.std(raw[:, 1]))
                                mean_critical = str(np.mean(raw[:, 2]))
                                sum_critical = str(np.sum(raw[:, 2]))
                                std_critical = str(np.std(raw[:, 2]))
                                mean_fata = str(np.mean(raw[:, 3]))
                                sum_fata = str(np.sum(raw[:, 3]))
                                std_fata = str(np.std(raw[:, 3]))
                                mean_tot = str(np.mean(raw))
                                sum_tot = str(np.sum(raw))
                                std_tot = str(np.std(raw))

                                # Write in file:
                                # Make a list of informations:
                                str_lst = []
                                str_lst.append(sum_tot)
                                for item in param_seir:
                                    str_lst.append(item)
                                for item in h_param:
                                    str_lst.append(item)
                                str_lst.append(mean_tot)
                                str_lst.append(sum_tot)
                                str_lst.append(std_tot)
                                str_lst.append(mean_test)
                                str_lst.append(sum_test)
                                str_lst.append(std_test)
                                str_lst.append(mean_hospit)
                                str_lst.append(sum_hospit)
                                str_lst.append(std_hospit)
                                str_lst.append(mean_critical)
                                str_lst.append(sum_critical)
                                str_lst.append(std_critical)
                                str_lst.append(mean_fata)
                                str_lst.append(sum_fata)
                                str_lst.append(std_fata)

                                convert_str_lst = []
                                for i in range(0, len(str_lst)):
                                    convert_str_lst.append(str(str_lst[i]))

                                print(convert_str_lst)

                                final_str = ';'.join(convert_str_lst)

                                file = open(
                                    'mod_select_result_part{}-{}-{}.csv'.format(select_part, self.smoothing,
                                                                                self.model.optimizer), "a")
                                file.write(final_str)
                                file.write('\n')

                                file.close()







if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='blabla')

    # smooth: 0 = non, 1 = oui
    parser.add_argument('--smooth', default=0)
    # Parties 1 à 4
    parser.add_argument('--part', default=1)
    # otptimizer
    parser.add_argument('--opti', default='LBFGSB')

    args = parser.parse_args()

    if args.smooth=='1':
        selector = model_selection(smooth=True)
    else:
        selector = model_selection(smooth=False)

    selected = 0
    if args.part == '1':
        selected = 1
    if args.part == '2':
        selected = 2


    selector.perform(select_part=selected, optimizer=args.opti)