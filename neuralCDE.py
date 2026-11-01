import jax
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")  # Ensure running on CPU
jax.config.update("jax_platform_name", "cuda")
# jax.config.update('jax_num_cpu_devices', 128)  # Set the number of CPU devices to 128
print(jax.devices())


import time
import numpy as np
import matplotlib.pyplot as plt
import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax.nn as jnn
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import optax  # https://github.com/deepmind/optax
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

plt.rcdefaults()
# figure styling
font=16
font_axis=20
plt.rcParams.update({'font.size': font})  # Set the desired font size
# plt.rcParams['text.usetex'] = True
plt.rcParams['figure.facecolor'] = 'white'  # Background color for figures
plt.rcParams['axes.facecolor'] = 'white'    # Background color for axes

class StateSpaceFunc(eqx.Module):
    mlp: eqx.nn.MLP
    stress_dim: int  # number of observable states
    k_inv: int       # number of hidden states
    input_size: int 
    output_size: int
    width_size: int 
    num_layers: int
    
    def __init__(self, stress_dim, k_inv, width_size, num_layers, *, key, **kwargs):
        super().__init__(**kwargs)
        self.stress_dim = stress_dim        # Number of observable states
        self.k_inv = k_inv                  # Number of hidden states
        self.input_size = 2*stress_dim + k_inv
        self.output_size = (((stress_dim + 1) * stress_dim)  // 2) + (k_inv * stress_dim) # Size for symmetric matrix + hidden state interactions
        self.width_size = width_size
        self.num_layers = num_layers

        self.mlp = eqx.nn.MLP(
            in_size = self.input_size,
            out_size = self.output_size,
            width_size = self.width_size,
            depth = self.num_layers + 1,
            activation = jnn.selu,          # softplus, relu, selu, gelu, tanh
            key = key,
        )

    def construct_jacobian(self, output):
        """Constructs the structured Jacobian matrix from neural network output"""
        stress_dim, k_inv = self.stress_dim, self.k_inv
        
        # Split the output into symmetric part and hidden state interactions
        sym_size = (stress_dim + 1) * stress_dim // 2
        sym_part = output[:sym_size]
        hidden_part = output[sym_size:]
        
        # Create indices for upper triangular part
        i_indices, j_indices = jnp.triu_indices(stress_dim)
        
        # Initialize matrix with zeros
        sym_matrix = jnp.zeros((stress_dim, stress_dim))
        
        # Set upper triangular values
        sym_matrix = sym_matrix.at[i_indices, j_indices].set(sym_part)
        
        # Set lower triangular values (transpose to make symmetric)
        sym_matrix = sym_matrix + jnp.triu(sym_matrix, k=1).T
    
        # Construct full Jacobian
        jac_matrix = jnp.zeros((stress_dim + k_inv, stress_dim))
        # Set symmetric part
        jac_matrix = jac_matrix.at[:stress_dim, :stress_dim].set(sym_matrix)
        # Set hidden state interactions
        hidden_matrix = hidden_part.reshape(k_inv, stress_dim)
        jac_matrix = jac_matrix.at[stress_dim:, :stress_dim].set(hidden_matrix)
        
        return jac_matrix
    
    def __call__(self, t, y, args):
        """forward call"""
        interp = args["control"]
        strain=interp.evaluate(t)
        dstrain=interp.derivative(t)
    
        # Get network output and construct Jacobian y(o) is the time
        input = jnp.concatenate([y[1:], dstrain[1:]])  # consider : jnp.sign and d_epsilon - note that the first element is time thats why its excluded
        output = self.mlp(input)
        # output =  jnp.sign(output) * ((jnp.exp(jnp.abs(output)))-1) 
        jacobian = self.construct_jacobian(output)
   
        
        m, n = jacobian.shape
        # Create a new matrix with dimensions (m+1, n+1) to add time dimension
        f = jnp.zeros((m + 1, n + 1))
        # Set the top-left corner to 1
        f = f.at[0,0].set(1)
        # Place the original matrix F in the bottom right corner
        f = f.at[1:, 1:].set(jacobian)

        return f

class StateSpaceNeuralCDE(eqx.Module):
    # initial: eqx.nn.MLP
    func: StateSpaceFunc
    # linear: eqx.nn.Linear

    def __init__(self, stress_dim, k_inv, width_size, num_layers, *, key, **kwargs):
        super().__init__(**kwargs)
        ikey, fkey, lkey = jr.split(key, 3)
        # self.initial = eqx.nn.MLP(stress_dim, stress_dim + k_inv, width_size, num_layers, key=ikey)     # add this feature later: this for initial conditions 
        self.func = StateSpaceFunc(stress_dim, k_inv, width_size, num_layers, key=fkey)
    
    def __call__(self, ts, tc, y0, coeffs):
        # Each sample of data consists of some time steps `ts`, and some `coeffs`
        # parameterising a control path. These are used to produce a continuous-time
        # input path `control`.
        
        dt = ts[1]-ts[0]

        control = diffrax.CubicInterpolation(tc, coeffs)
        strain = {"control": control, "ts": ts, "dt": dt}  # Create a dictionary to pass to the function
        term = diffrax.ControlTerm(self.func, control).to_ode()

        # y0 = self.initial(control.evaluate(ts[0]))
        # y0 = jnp.concatenate([ts[0],y0], axis=-1)  # Concatenate along the last dimension
        # dt0 = None
        # y0 = self.initial(control.evaluate(ts[0]))

        # below is the arrangment imipicit solver
        # root_finder = optx.Newton(rtol=1e-8, atol=1e-8)
        # solver = diffrax.ImplicitEuler(root_finder=root_finder)
    
        # below is the arrangment explicit solver
        solver = diffrax.Euler()   # Euler(), Tsit5(), Dopri5()
    
        # adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=100)
        saveat = diffrax.SaveAt(ts=ts, solver_state=True)

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0 = ts[0],
            t1 = ts[-1],
            dt0 = ts[1] - ts[0],
            y0 = y0,
            args=strain,
            # stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-6),
            saveat = saveat,
            # max_steps=10000,
            # adjoint=adjoint,
        )
        return solution.ys

def interpolate_trajectories(time, data, factor=1):
    """
    Interpolates each trajectory in the data to a new time resolution.
    
    Parameters:
    - data: np.ndarray of shape (N_trajectories, T_original, 1)
    - factor: float, >1 for upsampling, <1 for downsampling
    
    Returns:
    - new_data: np.ndarray of shape (N_trajectories, T_new, 1)
    """
    tend= time[-1]  # Get the last time point
    n_traj, n_t, c_comp = data.shape
    new_n_t = int(n_t * factor)
    new_data = np.zeros((n_traj, new_n_t, c_comp))

    for i in range(n_traj):
        for j in range(c_comp):
            # Interpolate each component of the trajectory
            new_x = np.linspace(0, tend, new_n_t)
            interp = interp1d(time, data[i, :, j], kind='cubic')
            new_data[i, :, j] = interp(new_x)

    return new_data

def loss(prediction, y_true):
    """loss types"""
    # loss = (jnp.mean((prediction - y_true) ** 2))               # MSE loss
    loss = jnp.sqrt(jnp.mean((prediction - y_true) ** 2))     # RMSE loss
    #loss = (jnp.mean(jnp.abs(prediction - y_true)))           # MAE loss 
      
    return loss

@eqx.filter_value_and_grad
def grad_loss(model, data, stress_dim, k_inv):

    ti, tc, yi, *coeff_i = data
    ti = ti.squeeze(-1)
    tc = tc.squeeze(-1)
    
    y0 = yi[:, 0]
    zeros = jnp.zeros((y0.shape[0], k_inv))              # Create an array of zeros with shape e.g., (1000,1,3)
    y0_expanded = jnp.concatenate([y0, zeros], axis=-1)  # Concatenate along the last dimension

    y_pred = eqx.filter_vmap(model, in_axes=(0, 0, 0, 0))(ti, tc, y0_expanded, coeff_i)

    return loss(y_pred[:, :, 1:stress_dim+1], yi[:, :, 1:stress_dim+1])     # 1:stress_dim+1

@eqx.filter_jit(donate="all")
def train_step(model, data, opt_state, optim, stress_dim, k_inv):

    loss, grads = grad_loss(model, data, stress_dim, k_inv)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return loss, model, opt_state

@eqx.filter_jit(donate="all-except-first")
def evaluate(model, data, stress_dim, k_inv, scale_out):

    ti, tc, yi, *coeff_i = data
    ti = ti.squeeze(-1)
    tc = tc.squeeze(-1)
    
    y0 = yi[:, 0]
    zeros = jnp.zeros((y0.shape[0], k_inv))                 # Create an array of zeros with shape e.g., (1000,1,3)
    y0_expanded = jnp.concatenate([y0, zeros], axis=-1)     # Concatenate along the last dimension

    y_pred = eqx.filter_vmap(model, in_axes=(0, 0, 0, 0))(ti, tc, y0_expanded, coeff_i)

    return loss(y_pred[:, :, 1:stress_dim+1], yi[:, :, 1:stress_dim+1]), jnp.mean(jnp.abs(y_pred[:, :, 1:stress_dim+1] - yi[:, :, 1:stress_dim+1]) * scale_out ), jnp.std( jnp.abs(y_pred[:, :, 1:stress_dim+1] - yi[:, :, 1:stress_dim+1]) * scale_out )     # 1:stress_dim+1

def dataloader(arrays, batch_size):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    
    while True:  # Infinite loop to keep yielding batches
        start = 0
        end = batch_size
        while end <= dataset_size:  # Changed to <= to include final complete batch
            batch_indices = jnp.arange(start, end)
            yield tuple(array[batch_indices] for array in arrays)
            start = end
            end = start + batch_size
        
        # Optional: handle final incomplete batch
        if start < dataset_size:
            batch_indices = jnp.arange(start, dataset_size)
            yield tuple(array[batch_indices] for array in arrays)

def train_model(print_every, model, training_data, validation_data, scale_out, stress_dim, k_inv, batch_size, lr_strategy, steps_strategy, length_strategy):
    
    # Exponential decay of learning rate: initial_lr * decay_rate ** (step / decay_steps)
    schedule = optax.exponential_decay(
       init_value=lr_strategy,
       transition_steps=1000,
       decay_rate=0.9,
       staircase=True  # or False for smooth decay
    )
    optim = optax.adam(schedule)

    # optim = optax.adam(lr_strategy)

    # optim = optax.chain(
    # optax.clip_by_global_norm(1.0),   # Clip gradients 1.0-5.0
    # optax.adam(schedule)
    # )
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    train_losses = []
    validation_losses = []
    mean_errors = []
    std_errors = []

    batch_num = len(range(0, training_data[0].shape[0], batch_size))
    
    for step in range(steps_strategy):
        epoch_train_loss = 0.0

        t_start = time.time()
        for i, data_batch in zip(range(batch_num), dataloader(training_data, batch_size)):
 
            loss_train, model, opt_state = train_step(model, data_batch, opt_state, optim, stress_dim, k_inv)

            epoch_train_loss += loss_train.item() / batch_num
        
        t_end = time.time()
        
        train_losses.append(epoch_train_loss)

        if (step % print_every) == 0 or step == steps_strategy - 1:
                print(f"Step: {step + 1}, Training Loss: {epoch_train_loss}, Epoch computation time: {t_end - t_start}")

        val_losses, mean_error, std_error = evaluate(model, validation_data, stress_dim, k_inv, scale_out)
        validation_losses.append(val_losses)    # scaled space
        mean_errors.append(mean_error/1e6)      # MPa
        std_errors.append(std_error/1e6)        # MPa
    
    return model, train_losses, validation_losses, mean_errors, std_errors

def main(
    batch_size = 120,
    train_sample = 2000,         # number of training samples
    val_sample = 100,            # number of validation samples
    lr_strategy = 1e-4,          # (1e-3, 1e-4),
    steps_strategy = 3000,       # (1000, 10000),
    length_strategy = 1,         # (1, 1),
    stress_dim = 6,              # observable states
    k_inv = 6,                   # number of internal state variables
    width_size = 100,
    num_layers = 510,              # consider that puting 4 means 3 layers with one input and one output
    seed=5678,
    plot=True,
    print_every=20,
):

    # import and navigate through the dataset
    # npy_time = np.load('CPFEtime.npy')
    # build time data
    n_trajectories = 2197
    n_timepoints = 201
    time_vector = np.linspace(0, 1, n_timepoints)  # shape (201,)
    npy_time = np.tile(time_vector, (n_trajectories, 1))[:, :, np.newaxis]  # shape (2197, 201, 1)
    strain = np.load('CPFEstrain.npy')
    stress = np.load('CPFEstress.npy')

    npy_time = np.delete(npy_time, 1113, axis=0)
    strain = np.delete(strain, 1113, axis=0)
    stress  = np.delete(stress, 1113, axis=0)

    # npy_time = npy_time[:,:,0:1]
    # strain = strain[:,:,0:1]
    # stress = stress[:,:,0:1]

    # # Initialize arrays for the new strain and stress tensors with 9 components
    # strain = np.zeros((output.shape[0], output.shape[1], 6))
    # stress = np.zeros((output.shape[0], output.shape[1], 6))

    # # Fill the strain tensor components
    # strain[:, :, 0] = output[:, :, 0]  # E11
    # strain[:, :, 1] = output[:, :, 1]  # E22
    # strain[:, :, 2] = output[:, :, 2]  # E33
    # strain[:, :, 3] = output[:, :, 3]  # E12 
    # strain[:, :, 4] = output[:, :, 4]  # E13
    # strain[:, :, 5] = output[:, :, 5]  # E23

    # # Fill the stress tensor components
    # stress[:, :, 0] = output[:, :, 6] * 1e6   # S11
    # stress[:, :, 1] = output[:, :, 7] * 1e6  # S22
    # stress[:, :, 2] = output[:, :, 8] * 1e6  # S33
    # stress[:, :, 3] = output[:, :, 9] * 1e6  # S12 
    # stress[:, :, 4] = output[:, :, 10] * 1e6  # S13
    # stress[:, :, 5] = output[:, :, 11] * 1e6 # S23

    # Importing and Scaling data
    # npy_time = np.expand_dims(np.load('t.npy'), axis=-1)
    # strain = np.expand_dims(np.load('strain.npy'), axis=-1)
    # stress = np.expand_dims(np.load('S.npy'), axis=-1)
    # state = np.expand_dims(np.load('state.npy'), axis=-1) # z in the paper
    # full_state = np.concatenate([stress, state], axis=-1)
    
    val_idx = train_sample + val_sample

    time_train = npy_time[:train_sample]
    time_val = npy_time[train_sample:val_idx]
    time_test = npy_time[val_idx:]

    strain_train = strain[:train_sample]
    strain_val = strain[train_sample:val_idx]
    strain_test = strain[val_idx:]

    stress_train = stress[:train_sample]
    stress_val = stress[train_sample:val_idx]
    stress_test = stress[val_idx:]

    # Scaling data
    max_abs_strain = np.abs(strain_train[:,:,:6]).max()
    strain_train_scaled = strain_train[:,:,:6] / max_abs_strain

    max_abs_stress = np.abs(stress_train[:,:,:6]).max()
    stress_train_scaled = stress_train[:,:,:6] / max_abs_stress
    
    scaler_E = max_abs_strain
    scaler_S = max_abs_stress
    stress_scale = scaler_S


    # Assuming time, strain, stress are your original arrays:
    time_enter = time_train[0,:,0]  # Use the first trajectory's time for interpolation
    time_interp_train   = jax.device_put(interpolate_trajectories(time_enter, time_train))
    strain_interp_train = jax.device_put(interpolate_trajectories(time_enter, strain_train_scaled))

    jax_time_train = jax.device_put(time_train)
    jax_strain_train = jax.device_put(strain_train_scaled)
    jax_stress_train = jax.device_put(stress_train_scaled)
    # jax_state = jax.device_put(full_state_train_scaled)

    x_strain_train = jnp.concatenate([jax_time_train, jax_strain_train], axis=-1)   # time is a channel
    coeffs_train = jax.vmap(diffrax.backward_hermite_coefficients)(jax_time_train.squeeze(-1), x_strain_train)
    coeffs_train = tuple(arr[:, 1:, :] for arr in coeffs_train)

    t_train = jax_time_train[:,:-1]
    t_c_train = time_interp_train[:,:-1]
    # ys_train = jnp.concatenate([jax_time_train, jax_state_train], axis=-1)
    y_train = jnp.concatenate([jax_time_train[:,:-1], jax_stress_train[:,:-1]], axis=-1)

    strain_val_scaled = strain_val[:,:,:6] / max_abs_strain
    stress_val_scaled = stress_val[:,:,:6] / max_abs_stress

    # Assuming time, strain, stress are your original arrays:
    time_enter = time_val[0,:,0]  # Use the first trajectory's time for interpolation
    time_interp_val   = jax.device_put(interpolate_trajectories(time_enter, time_val))
    strain_interp_val = jax.device_put(interpolate_trajectories(time_enter, strain_val_scaled))

    jax_time_val = jax.device_put(time_val)
    jax_strain_val = jax.device_put(strain_val_scaled)
    jax_stress_val = jax.device_put(stress_val_scaled)
    # jax_state = jax.device_put(full_state_val_scaled)

    x_strain_val = jnp.concatenate([jax_time_val, jax_strain_val], axis=-1)   # time is a channel
    coeffs_val = jax.vmap(diffrax.backward_hermite_coefficients)(jax_time_val.squeeze(-1), x_strain_val)
    coeffs_val = tuple(arr[:, 1:, :] for arr in coeffs_val)

    t_val = jax_time_val[:,:-1]
    t_c_val = time_interp_val[:,:-1]
    # ys_val = jnp.concatenate([jax_time_val, jax_state_val], axis=-1)
    y_val = jnp.concatenate([jax_time_val[:,:-1], jax_stress_val[:,:-1]], axis=-1)

    # Defining model
    key = jr.PRNGKey(seed)
    train_data_key, test_data_key, model_key, loader_key = jr.split(key, 4)

    model = StateSpaceNeuralCDE (stress_dim, k_inv, width_size, num_layers, key = model_key)
    model = eqx.tree_deserialise_leaves("model.eqx", model)

    # training_data = DataLoader((t_train, y_train) + coeffs_train, batch_size=batch_size, shuffle=False, collate_fn=no_tensor_collate)
    training_data = (t_train, t_c_train, y_train) + coeffs_train
    validation_data = (t_val, t_c_val, y_val) + coeffs_val

    # Training loop
    model, train_losses, validation_losses, mean_error, std_error = train_model(print_every, model, training_data, validation_data, stress_scale, stress_dim, k_inv, batch_size, lr_strategy, steps_strategy, length_strategy)

    return model, train_losses, validation_losses, mean_error, std_error, scaler_E, scaler_S

tstart = time.time()
model, train_losses, validation_losses, mean_errors, std_errors, scaler_E, scaler_S = main()
tend = time.time()

print(f"Total computation time: {tend - tstart}")

scales = (scaler_E, scaler_S)
# Save the tuple
np.save('scales.npy', scales)

# Save the model to a file.
eqx.tree_serialise_leaves("model.eqx", model)
# Save the results to a file
jnp.save('loss_results.npy', (train_losses, validation_losses, mean_errors, std_errors))

plt.figure(figsize=(5, 4))
plt.plot(train_losses , label='Training Loss', color='blue', linestyle='--')
plt.plot(validation_losses , label='Validation Loss', color='red')
plt.xlabel('Epoch', fontsize=font, fontname='Times New Roman')
plt.ylabel('RMSE Loss', fontsize=font, fontname='Times New Roman')
plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='major', direction='in')
plt.tick_params(axis='both', which='minor', direction='in')
# plt.autoscale(tight=True)
plt.legend(prop={'family': 'Times New Roman', 'size': 14}, frameon=False)
plt.ylim(1e-6, 1)
plt.grid(True, linestyle='--')
plt.tight_layout()
# Save the figure as a PDF
plt.savefig("Training_loss.pdf", format="pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(5, 4))
plt.plot(mean_errors , color='blue')
plt.plot(mean_errors, label='Mean', color='blue', alpha=0.7)
# plt.plot(np.array(mean_errors) + 3* np.array(std_errors) , '.', label='Mean + 3 std', color='red', alpha=0.2)
# plt.plot(mean_errors - std_errors , color='red', linestyle='--')
# plt.plot(mean_errors + std_errors , color='red', linestyle='--')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-2, 1000)
plt.xlabel('Epoch', fontsize=font, fontname='Times New Roman')
plt.ylabel('Estimation error (MPa)', fontsize=font, fontname='Times New Roman')
plt.tick_params(axis='both', which='major', direction='in')
plt.tick_params(axis='both', which='minor', direction='in')
plt.grid(True, linestyle='--')
# plt.autoscale(tight=True)
# plt.legend(prop={'family': 'Times New Roman', 'size': 14}, frameon=False)
plt.tight_layout()
# Save the figure as a PDF
plt.savefig("MPa_loss.pdf", format="pdf", bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------
# Sava the scale
np.savetxt('scales.dat', scales)

# Sava the model
mlp_model = model.func.mlp
filename = 'mlp_parameters.dat'

# Get all parameters as a pytree
params = eqx.filter(model, eqx.is_array)

# Flatten the pytree to get parameter names and values
flat_params, tree_def = jax.tree_util.tree_flatten_with_path(params)
# print(flat_params)

def count_params(model):
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))

n_params = count_params(mlp_model)
print("Number of parameters:", n_params)

# Open the file in write mode
with open(filename, 'w') as f:
    for key_path, value in flat_params:
        # Create a readable parameter name from the path
        param_name = '.'.join([str(k.key) if hasattr(k, 'key') else str(k) for k in key_path])
        
        # Write the parameter name and shape
        f.write(f"{param_name} {value.shape}\n")
        
        # Write the flattened parameter values
        flattened_values = np.array(value).flatten()
        np.savetxt(f, flattened_values, fmt='%f')

