# encoding:utf-8
import random
import math


class EM_for_coin:
    def __init__(self, rolls, thetaA=None, thetaB=None, deviation=0.1):
        self.thetaA = thetaA or random.random()
        self.thetaB = thetaB or random.random()
        self.rolls = rolls
        self.deviation = deviation

    """
    Use existing estimates of hidden variables to calculate their maximum likelihood estimates
    """

    def E_Step(self):
        H_A, H_B, T_A, T_B = 0, 0, 0, 0
        for trial in self.rolls:
            H_num = trial.count("H")
            T_num = trial.count("T")
            P_A, P_B = self.estimate_likelihood(H_num, T_num)
            H_A += H_num * P_A
            T_A += T_num * P_A
            H_B += H_num * P_B
            T_B += T_num * P_B
        return H_A, T_A, H_B, T_B

    """
    Use the maximum likelihood value obtained by E-step to calculate the value of the parameter 
    """

    def M_Step(self, H_A, T_A, H_B, T_B):
        new_thetaA = H_A/(H_A+T_A)
        new_thetaB = H_B/(H_B+T_B)
        return new_thetaA, new_thetaB

    """
    According to Baye's theorem and the law of total probability P(A)=P(B)
    we can partition all of the events probability
    """

    def estimate_likelihood(self, H_num, T_num):
        P_events_on_A = math.pow(self.thetaA, H_num) * \
            math.pow(1 - self.thetaA, T_num)
        P_events_on_B = math.pow(self.thetaB, H_num) * \
            math.pow(1 - self.thetaB, T_num)
        P_A = P_events_on_A/(P_events_on_A+P_events_on_B)
        P_B = P_events_on_B/(P_events_on_A+P_events_on_B)
        return P_A, P_B

    """
    Use iterative methods to continuously approach the true value 
    until the difference between theta before and after is less than our requirement
    """

    def run(self):
        while True:
            H_A, T_A, H_B, T_B = self.E_Step()
            new_thetaA, new_thetaB = self.M_Step(H_A, T_A, H_B, T_B)
            if abs(self.thetaA-new_thetaA) < self.deviation and abs(self.thetaB-new_thetaB) < self.deviation:
                break
            else:
                print("thetaA:%.10f thetaB:%.10f" % (self.thetaA, self.thetaB))
                self.thetaA = new_thetaA
                self.thetaB = new_thetaB
        return self.thetaA, self.thetaB


if __name__ == "__main__":
    rolls = input("Please enter multiple results of tossing a coin as a list:\n \
    (e.g HTTTHHTHTH, HHHHTHHHHH, HTHHHHHTHH, HTHTTTHHTT, THHHTHHHTH)\n").split(',')
    thetaA, thetaB = EM_for_coin(rolls, 0.6, 0.5, 0.0001).run()
    print("thetaA:%.10f thetaB:%.10f" % (thetaA, thetaB))
