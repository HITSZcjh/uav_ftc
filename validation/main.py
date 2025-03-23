import onnxruntime
import numpy as np
import os
from uav_model import uav_model, quaternion_to_rotation_matrix
import matplotlib.pyplot as plt
from high_level_controller import LPF, PositionController, PrimaryAxisAttitudeController

if __name__=="__main__":

    # Load ONNX model for the RL-based controller
    session = onnxruntime.InferenceSession(os.path.dirname(os.path.realpath(__file__)) + "/validation.onnx", providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    ts = 0.0025  # Simulation time step
    N = 8        # Control update interval multiplier for High level controller
    model = uav_model(ts)  # UAV model

    # High level controller
    pos_controller = PositionController(ts*N)  
    axis_controller = PrimaryAxisAttitudeController(ts*N)  

    pos_desired = np.array([1, 1, 3], dtype=np.float32)     # Desired position (change this one)
    pos_rl_goal = np.array([0, 0, 3], dtype=np.float32)     # RL goal position (no change)

    state = np.zeros(27)  # Input state vector to the RL controller
    u = 0 * np.ones(4)    # Initial rotor commands
    u_lpf = LPF(ts, 0.025)          # Low-pass filter for rotor commands
    omega_dot_lpf = LPF(ts, 0.12)   # Low-pass filter for angular acceleration
    model.k = np.array([1, 1, 1, 1], dtype=np.float32)  # Initial motor efficiency coefficients

    for i in range(3000):  # Simulation loop
        # Introduce rotor failure at step 1000 (second motor fails)
        if i > 1000:
            model.k = np.array([1, 0, 1, 1], dtype=np.float32)

        # Get UAV sensor observations
        p, v, q, w, acc_B = model.get_obs()  # Position, velocity, quaternion, angular velocity, body-frame acceleration(IMU)
        R = quaternion_to_rotation_matrix(q)

        # Update high-level controller every N steps
        if i % N == 0:
            acc_I_des, n_des_I, n_des_I_dot = pos_controller.calc(pos_desired, p, v)
            p_des, q_des, acc_z_des = axis_controller.calc(R, w[2], acc_I_des, n_des_I, n_des_I_dot)

        # Prepare the input state for the RL controller
        state[:13] = np.hstack((p, v, q, w))  
        state[0:3] += pos_rl_goal - pos_desired  # Adjust position based on RL goal offset
        state[13:17] = u_lpf.calc(u)  # Filtered control input
        state[17:20] = acc_B
        # High level command          
        state[20] = p_des             
        state[21] = q_des             
        state[22] = 0                 
        state[23] = acc_z_des         
        state[24:27] = omega_dot_lpf.calc_derivative(w)  # Angular acceleration estimate

        # Normalize input features
        state /= np.array((5, 5, 5,       # position
                           5, 5, 5,       # velocity
                           1, 1, 1, 1,    # quaternion
                           5, 5, 5,       # angular velocity
                           6, 6, 6, 6,    # rotor commands
                           5, 5, 20,      # acc_B
                           5, 5, 5, 20,   # High level command
                           50, 50, 50),   # angular acceleration
                           dtype=np.float32)

        # Run RL controller inference
        outputs = session.run(output_names, {input_name: state.astype(np.float32).reshape(1, 27)})

        # Parse and scale the output action
        action = np.array(outputs, dtype=np.float32).squeeze()
        action = action * 4 + 2  # Rescale action from [-1, 1] to [-2, 6]
        u = np.clip(action, 0, 6)  # Clip to actuator limits

        # Apply control to UAV model
        model.step(u)

    # Show simulation result
    model.log_show()
    plt.show()
