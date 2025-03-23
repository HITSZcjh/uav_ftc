import numpy as np


BW = 50
class LPF:
    def __init__(self, ts, tau):
        self.ts = ts  
        self.tau = tau  
        self.last_input = None
        self.last_output = None

    def calc(self, input):
        if self.last_output is None:
            self.last_output = input  
        output = (self.ts * input + self.tau * self.last_output) / (self.tau + self.ts)
        self.last_output = output
        return output

    def calc_with_derivative(self, input):
        if self.last_input is None:
            self.last_input = input  
        if self.last_output is None:
            self.last_output = input  
        
        output = (self.ts * input + self.tau * self.last_output) / (self.tau + self.ts)
        derivative = (input - self.last_input + self.tau * self.last_output) / (self.tau + self.ts)
        
        self.last_input = input
        self.last_output = output
        
        return output, derivative

    def calc_derivative(self, input):
        if self.last_input is None:
            self.last_input = input  
        if self.last_output is None:
            self.last_output = input  

        output = (input - self.last_input + self.tau * self.last_output) / (self.tau + self.ts)
        
        self.last_input = input
        self.last_output = output
        
        return output

class PositionController(object):
    def __init__(self, ts) -> None:
        self.ts = ts

        self.kp_pos = np.array([1,1,1], dtype=np.float32)
        self.kp_vel = np.array([2,2,6], dtype=np.float32)

        self.int_lim = 5.0
        self.max_vel = 10.0
        self.max_angle = 10./57.3
        self.max_lateral = abs(9.81*np.tan(self.max_angle))

        self.g = np.array([0,0,-9.81], dtype=np.float32)
        self.last_n_des_I = np.array([0,0,1], dtype=np.float32)

    def calc(self, pos_target, pos_real, vel_real):
        pos_err = pos_target - pos_real
        vel_target = self.kp_pos * pos_err
        vel_target = np.clip(vel_target, -self.max_vel, self.max_vel)
        vel_err = vel_target - vel_real
        acc_I_des = self.kp_vel * vel_err
        

        lat_ratio = np.linalg.norm(acc_I_des[:2])/self.max_lateral
        if lat_ratio > 1:
            acc_I_des[:2] /= lat_ratio
        
        acc_I_des[2] = np.clip(acc_I_des[2], -5, 5)
        n_des_I = (acc_I_des-self.g)/np.linalg.norm(acc_I_des-self.g)
        n_des_I_dot = (n_des_I - self.last_n_des_I) / self.ts
        self.last_n_des_I = n_des_I
        n_des_I_dot = np.clip(n_des_I_dot, -0.5, 0.5)

        return acc_I_des, n_des_I, n_des_I_dot
    
class PrimaryAxisAttitudeController(object):
    def __init__(self, ts) -> None:
        self.ts = ts
        self.g = np.array([0,0,-9.81], dtype=np.float32)

        self.n_B = np.array([0.0,0.0,1.0], dtype=np.float32)

        self.kx = 5
        self.ky = 5
        self.p_des_lpf = LPF(self.ts, 1.0/BW)
        self.q_des_lpf = LPF(self.ts, 1.0/BW)
        self.acc_z_des_lpf = LPF(self.ts, 1.0/BW)
    def calc(self, R, r, acc_I_des, n_des_I, n_des_I_dot):
        n_des_B = R.T@n_des_I
        h1 = n_des_B[0]
        h2 = n_des_B[1]
        h3 = n_des_B[2]
        n_B_x = self.n_B[0]
        n_B_y = self.n_B[1]
        n_B_z = self.n_B[2]
        vout = np.array([self.kx*(n_B_x-h1),self.ky*(n_B_y-h2)], dtype=np.float32)
        temp = np.array([[0,1.0/h3],[-1.0/h3,0]], dtype=np.float32)
        n_des_I_hat_dot = (R.T@n_des_I_dot)[:2]
        temp1 = temp@(vout-r*np.array([h2,-h1], dtype=np.float32)-n_des_I_hat_dot)
        p_des = temp1[0]
        q_des = temp1[1]
        p_des = self.p_des_lpf.calc(p_des)
        q_des = self.q_des_lpf.calc(q_des)
        p_des = np.clip(p_des,-10,10)
        q_des = np.clip(q_des,-10,10)

        acc_z_des = np.linalg.norm(acc_I_des-self.g)/n_B_z
        acc_z_des = self.acc_z_des_lpf.calc(acc_z_des)
        acc_z_des = np.clip(acc_z_des,-50,50)

        return p_des, q_des, acc_z_des

