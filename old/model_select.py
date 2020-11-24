import numpy as np
import matplotlib.pyplot as plt
import SEIR
import argparse


class model_selection():

    def __init__(self, smooth=False):

        # ========================================== #
        #           Hyper-Parameters values to test
        # ========================================== #

        self.w_1 = [1, 2, 3]
        self.w_2 = [1, 2, 3]
        self.w_3 = [1, 2, 3]
        self.w_4 = [1, 2, 3]
        self.w_5 = [1, 2, 3]
        self.w_6 = [1, 2, 3]

        self.smoothing = smooth

        # Binomial smoother
        self.b_s_1 = [1, 2, 3, 4, 5, 6, 7, 8]
        self.b_s_2 = [1, 2, 3, 4, 5, 6, 7, 8]
        self.b_s_3 = [1, 2, 3, 4, 5, 6, 7, 8]
        self.b_s_4 = [1, 2, 3, 4, 5, 6, 7, 8]
        self.b_s_5 = [1, 2, 3, 4, 5, 6, 7, 8]
        self.b_s_6 = [1, 2, 3, 4, 5, 6, 7, 8]

        # Optimizer to choose: 0 = Cobyla, 1 = LBFGSB, 2 = Auto
        self.optimizer = ['COBYLA', 'LBFGSB', 'AUTO']

        # Spare in parts
        self.part = [[1, 2], [3, 4], [5, 6], [7, 8]]

        self.model = SEIR.SEIR()
        if self.smoothing:
            self.model.smoothing = True
        self.model.import_dataset()
        self.model.fit_type = 'type_1'

    def perform(self, select_part=1, smooth=False):

        iter = 0
        total_iter = 2*6*6*6*6*6 * len(self.b_s_1)*6 *3

        for w_1 in self.part[select_part]:
            for w_2 in self.w_2:
                for w_3 in self.w_3:
                    for w_4 in self.w_4:
                        for w_5 in self.w_5:
                            for w_6 in self.w_6:
                                for b_s_1 in self.b_s_1:
                                    for b_s_2 in self.b_s_2:
                                        for b_s_3 in self.b_s_3:
                                            for b_s_4 in self.b_s_4:
                                                for b_s_5 in self.b_s_5:
                                                    for b_s_6 in self.b_s_6:
                                                        for optzr in self.optimizer:

                                                            iter += 1
                                                            print('iter {} / {}'.format(iter, total_iter))

                                                            self.model.w_1 = w_1
                                                            self.model.w_2 = w_2
                                                            self.model.w_3 = w_3
                                                            self.model.w_4 = w_4
                                                            self.model.w_5 = w_5
                                                            self.model.w_6 = w_6
                                                            self.model.b_s_1 = b_s_1
                                                            self.model.b_s_2 = b_s_2
                                                            self.model.b_s_3 = b_s_3
                                                            self.model.b_s_4 = b_s_4
                                                            self.model.b_s_5 = b_s_5
                                                            self.model.b_s_6 = b_s_6
                                                            self.model.optimizer = optzr

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
                                                                'mod_select_result_part{}-{}.csv'.format(select_part, self.smoothing), "a")
                                                            file.write(final_str)
                                                            file.write('\n')

                                                            file.close()






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='blabla')

    # smooth: 0 = non, 1 = oui
    parser.add_argument('--smooth', default=0)
    # Parties 1 à 4
    parser.add_argument('--part', default=1)

    args = parser.parse_args()

    if args.smooth=='1':
        selector = model_selection(smooth=True)
    else:
        selector = model_selection(smooth=False)

    selected = 0
    if args.part == '2':
        selected = 1
    if args.part == '3':
        selected = 2
    if args.part == '4':
        selected = 3

    selector.perform(select_part=selected)