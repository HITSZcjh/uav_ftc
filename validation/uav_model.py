import numpy as np

def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix.
    Input: q - shape=(4,), format: [w, x, y, z]
    Output: 3x3 rotation matrix R
    """
    w, x, y, z = q

    R = np.array([
        [1 - 2 * (y**2 + z**2),     2 * (x*y - w*z),     2 * (x*z + w*y)],
        [2 * (x*y + w*z),     1 - 2 * (x**2 + z**2),     2 * (y*z - w*x)],
        [2 * (x*z - w*y),         2 * (y*z + w*x), 1 - 2 * (x**2 + y**2)]
    ])
    return R

def quaternion_derivative(q, w):
    """
    Compute the derivative of quaternion.
    q: quaternion [qw, qx, qy, qz]
    w: angular velocity vector [wx, wy, wz]
    return: dq/dt
    """
    qw, qx, qy, qz = q
    wx, wy, wz = w

    q_dot = 0.5 * np.array([
        -qx * wx - qy * wy - qz * wz,
         qw * wx + qy * wz - qz * wy,
         qw * wy - qx * wz + qz * wx,
         qw * wz + qx * wy - qy * wx
    ])
    return q_dot

class uav_model:
    def __init__(self, ts) -> None:
        """
        Initialize UAV model parameters
        """
        self.ts = ts  # time step
        self.g = np.array([0, 0, -9.81], dtype=np.float32)  # gravity vector
        self.rotor_time_constant = 0.025
        self.rotor_moment_constant = 0.016
        self.body_length = 0.125
        self.mass = 0.764
        self.inertia = np.array([[0.0036, 0, 0], [0, 0.0029, 0], [0, 0, 0.0053]], dtype=np.float32)
        self.lin_drag_factor = 0.1
        self.ang_drag_factor = 1.5e-4
        self.yaw_moment_factor = 0.09

        # Control allocation matrix
        self.alloc_mat = np.array([[0, self.body_length, 0, -self.body_length],
                                   [-self.body_length, 0, self.body_length, 0],
                                   [self.rotor_moment_constant, -self.rotor_moment_constant,
                                    self.rotor_moment_constant, -self.rotor_moment_constant],
                                   [1, 1, 1, 1]], dtype=np.float32)
        
        self.mass_inv = 1.0 / self.mass
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.rotor_time_constant_inv = 1.0 / self.rotor_time_constant

        # Initial state [p, v, q, w, f]
        self.state = np.array([0, 0, 3,   # position
                               0, 0, 0,   # velocity
                               1, 0, 0, 0,  # quaternion
                               0, 0, 0,   # angular velocity
                               0, 0, 0, 0], dtype=np.float32)  # rotor forces
        self.state_dot = np.zeros_like(self.state)
        self.action_range = [0.0, 6.0]
        self.k = np.ones(4, dtype=np.float32)

        self.log_state_list = []

    def ode(self, x, u):
        """
        UAV dynamics (ordinary differential equations)
        x: current state
        u: control input
        return: state derivative
        """
        v = x[3:6]
        q = x[6:10]
        w = x[10:13]
        f = x[13:17]

        p_dot = v

        torque_thrust = self.alloc_mat @ f
        thrust = np.array([0, 0, torque_thrust[3]], dtype=np.float32)
        lin_drag = -self.lin_drag_factor * v
        R = quaternion_to_rotation_matrix(q)
        v_dot = self.g + self.mass_inv * (R @ thrust + lin_drag)

        q_dot = quaternion_derivative(q, w)

        f_dot = self.rotor_time_constant_inv * (self.k * u - f)

        torque = torque_thrust[:3]
        torque_thrust_dot = self.alloc_mat @ f_dot
        torque[2] += self.yaw_moment_factor * torque_thrust_dot[2]
        ang_drag = -self.ang_drag_factor * np.sign(w) * w**2
        w_dot = self.inertia_inv @ (torque + ang_drag - np.cross(w, self.inertia @ w))

        return np.hstack((p_dot, v_dot, q_dot, w_dot, f_dot))
    
    def step(self, action):
        """
        Perform one simulation step
        """
        u = np.clip(action, self.action_range[0], self.action_range[1])
        self.state_dot = self.ode(self.state, u)
        self.state += self.state_dot * self.ts
        self.state[6:10] = self.state[6:10] / np.linalg.norm(self.state[6:10])  # normalize quaternion

        self.log_state_list.append(self.state.copy())

        return self.state
    
    def get_obs(self):
        """
        Get observation from current state
        return: position, velocity, quaternion, angular velocity, body-frame acceleration
        """
        p = self.state[:3]
        v = self.state[3:6]
        q = self.state[6:10]
        w = self.state[10:13]

        acc_I = self.state_dot[3:6] - self.g
        R = quaternion_to_rotation_matrix(q)
        acc_B = R.T @ acc_I

        return p, v, q, w, acc_B

    def log_show(self):
        """
        Plot and animate the logged state trajectory
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        t = np.arange(0, len(self.log_state_list) * self.ts, self.ts)
        self.log_state_list = np.array(self.log_state_list)

        # Plot state trajectories
        self.fig, self.axs = plt.subplots(2, 2)
        self.axs[0, 0].plot(t, self.log_state_list[:, 0], label="px")
        self.axs[0, 0].plot(t, self.log_state_list[:, 1], label="py")
        self.axs[0, 0].plot(t, self.log_state_list[:, 2], label="pz")
        self.axs[0, 0].legend()

        self.axs[0, 1].plot(t, self.log_state_list[:, 3], label="vx")
        self.axs[0, 1].plot(t, self.log_state_list[:, 4], label="vy")
        self.axs[0, 1].plot(t, self.log_state_list[:, 5], label="vz")
        self.axs[0, 1].legend()

        self.axs[1, 0].plot(t, self.log_state_list[:, 6], label="qw")
        self.axs[1, 0].plot(t, self.log_state_list[:, 7], label="qx")
        self.axs[1, 0].plot(t, self.log_state_list[:, 8], label="qy")
        self.axs[1, 0].plot(t, self.log_state_list[:, 9], label="qz")
        self.axs[1, 0].legend()

        self.axs[1, 1].plot(t, self.log_state_list[:, 10], label="wx")
        self.axs[1, 1].plot(t, self.log_state_list[:, 11], label="wy")
        self.axs[1, 1].plot(t, self.log_state_list[:, 12], label="wz")
        self.axs[1, 1].legend()

        # Animation of 3D attitude arrows
        self.fig = plt.figure()
        self.axs = self.fig.add_subplot(111, projection='3d')
        self.axs.set_xlim(-5, 5)
        self.axs.set_ylim(-5, 5)
        self.axs.set_zlim(0, 6)

        self.arrow_length = 1.0
        self.log_R_list = []
        N = int(0.02 / self.ts)
        self.log_state_list = self.log_state_list[::N]

        for i in range(self.log_state_list.shape[0]):
            w, x, y, z = self.log_state_list[i, 6:10]
            self.log_R_list.append(np.array([
                [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
            ]))

        self.log_R_list = np.array(self.log_R_list)

        self.x_arrow = self.axs.quiver(self.log_state_list[0, 0], self.log_state_list[0, 1], self.log_state_list[0, 2],
                                       self.arrow_length * self.log_R_list[0, 0, 0], self.arrow_length * self.log_R_list[0, 1, 0], self.arrow_length * self.log_R_list[0, 2, 0],
                                       color='r', label='X')
        self.y_arrow = self.axs.quiver(self.log_state_list[0, 0], self.log_state_list[0, 1], self.log_state_list[0, 2],
                                       self.arrow_length * self.log_R_list[0, 0, 1], self.arrow_length * self.log_R_list[0, 1, 1], self.arrow_length * self.log_R_list[0, 2, 1],
                                       color='g', label='Y')
        self.z_arrow = self.axs.quiver(self.log_state_list[0, 0], self.log_state_list[0, 1], self.log_state_list[0, 2],
                                       self.arrow_length * self.log_R_list[0, 0, 2], self.arrow_length * self.log_R_list[0, 1, 2], self.arrow_length * self.log_R_list[0, 2, 2],
                                       color='b', label='Z')

        self.ani = FuncAnimation(self.fig, self.update,
                                 frames=self.log_R_list.shape[0], interval=20, repeat=True)

    def update(self, frame):
        """
        Animation frame update function
        """
        self.x_arrow.remove()
        self.y_arrow.remove()
        self.z_arrow.remove()

        self.x_arrow = self.axs.quiver(self.log_state_list[frame, 0], self.log_state_list[frame, 1], self.log_state_list[frame, 2],
                                       self.arrow_length * self.log_R_list[frame, 0, 0], self.arrow_length * self.log_R_list[frame, 1, 0], self.arrow_length * self.log_R_list[frame, 2, 0],
                                       color='r', label='X')
        self.y_arrow = self.axs.quiver(self.log_state_list[frame, 0], self.log_state_list[frame, 1], self.log_state_list[frame, 2],
                                       self.arrow_length * self.log_R_list[frame, 0, 1], self.arrow_length * self.log_R_list[frame, 1, 1], self.arrow_length * self.log_R_list[frame, 2, 1],
                                       color='g', label='Y')
        self.z_arrow = self.axs.quiver(self.log_state_list[frame, 0], self.log_state_list[frame, 1], self.log_state_list[frame, 2],
                                       self.arrow_length * self.log_R_list[frame, 0, 2], self.arrow_length * self.log_R_list[frame, 1, 2], self.arrow_length * self.log_R_list[frame, 2, 2],
                                       color='b', label='Z')
