# weights and hparams are scattered across the code
Tree[T] = Tree[T] | List[T] | Dict[T] | Set[T] | T
Tree[T, d] = Tree of depth d with T as the type
`treemap(tree, f)`: if `f` is a function, recursively applys `f` to all elements of tree. If `f` is a dictionary, applies `f[k]` to branches and leaves with key `k`. Post-order in both cases. Actually, do `tree_i.map(f)`
recursive_map
`treesum(tree, weighted=True, weights: Tree = None, learnable=True, hierachial=True, return_tree=False)`: weighted sum of all elements of tree -- with weights also on branches if `hierachial` is True. `weights` default to all 1's. Will return a `Tree` with `(sum, children)` on each node if `return_tree` is True. Actually, do `tree_i.sum(...)`
Since treesum looks for exact matches, the `vision` modality may process a single Tensor while the `hearing` modality processes 3 tensors. Neural networks should assume their inputs are tree-structured unless confident otherwise 
Many objects are trees
Nodes can be repeated in a Tree, but cycles are not allowed

P = Param('name', initial=0, size=()), keeps track of timeseries but looks like a list from the outside, eg, you can .append(Tree[T]) to add a new element but keep all the performance advantages of tensors. You can also index non-existant slots to add new elements like `P[t] = x`. You do not have to start from 0. This makes it convenient to add params in the middle of training and still be able to store them all togethor. I reocmmend using the absolute simulation step ofset for all time ofsets. (But make sure to log a fps rate at each time offset so you can compensate when performing fps-aware training) P even optimizes storing multiple contiguous time series, eg, 0..1034, 54324..645352, ..., which is useful when dreams only happen at night and observations during the day.
HP = HParam = Param but with hparam tag
MP = MetaParam = Param but with meta tag

I need to look at my notes to understand if I've made any suggestions for how to do this since a month ago.

....
# TODO: clarify difference between Variable and MetaParam
# Add default value / default function for Variable indexes
#   Specify interpolation function for default function 
....


signal_forier_recon_loss = HP(name='signal_forier_recon_loss', initial=0, size=())

signal_forier_recon_loss[t] = sum(
    (signal_recon_loss[t] + signal_recon_loss[t-1]) / 2
    for t in range(1, T)
)

signal_forier_pred[t+1] = signal_forier_pred[t] + signal_forier_recon_loss[t]

# for now, I am writing code without thought about low-level tensor optimizations
# - timeseries are just in a Python list
# - not all indv tree-ops are written using treemap

# types
# not complete
# mainly for documentation purposes.
# but useful to distinguish properties of T_INPUT/ENC/DEC
# since they are not necesarily the same
T_INPUT = Tree[Tensor]
T_INPUT_ENC = Tree[Tensor] # k in T_INPUT_ENC -> k in T_IMMEDIATE
T_INPUT_DEC = Tree[Tensor] # k in T_INPUT_DEC -> k in T_INPUT
T_IMMEDIATE = Tree[Tensor]

# i/o
modalities[t] = List[Modality] # provides I/O information
encoders: List[Tuple[str, Callable]] # name, function
decoders = List[Tuple[str, Callable]] # name, function
input[t]: T_INPUT
reconstruction[t]: T_INPUT
encoding[t]: T_INPUT_ENC = each encoder's encoding (# encoders <> # inputs)
decoding[t]: T_INPUT_DEC = each decoder's decoding (# decoders <> # outputs)

# meta-signals

## controllable meta-signals slowly adapt via gradient descent
lr[t]: Tree[float, 2] # learning rate for each module for each layer
T_FFT[t] = 30 # however many timesteps back you need to get a reasonable FFT
T_IFFT_PREV[t] = 30 # how many timesteps to rollback the meta-signal IFFT
T_IFFT_FORE[t] = 30 # how many timesteps to rollforward the meta-signal IFFT

## uncontrollable meta-signals are hardcoded
surprise[t-1] = treesum(treemap(pred_immediate[t-2], immediate[t-1])) # cumulative surprise is just the root[0] value
activation_statistics[t-1]: Tree[Tensor, 2] = mean, std, skewness, kurtosis, etc for each layer for each module
gradient_statistics[t-1]: Tree[Tensor, 2] = mean, std, skewness, kurtosis, etc for each layer for each module
meta_signal_forier_modes[t-1] = treemap(meta_signals[t-1], lambda x: fft(x, n=3))

controllable_meta_signals = [T_FFT[t], lr[t]]
uncontrollable_meta_signals = {
  surprise[t-1], reward[t-1], 
  activation_statistics[t-1], gradient_statistics[t-1], activation_statistics_forier_modes[t-1], gradient_statistics_forier_modes[t-1],
  meta_signal_forier_modes[t-1]
}
meta_signals = [controllable_meta_signals, uncontrollable_meta_signals]

# internal state

top_down[t] = f_top_down(state[t-1])

immediate[t] = {
  'encodings': encodings[t],
  'meta_signals': meta_signals[t],
  'top_down': top_down[t],
}

stm: Multigraph

semantic_ltm: Multigraph
Episode = List[T_IMMEDIATE]
episodic_ltm: Set[Episode]

### Note: Previous LTM states are not saved
ltm = (semantic_ltm, episodic_ltm)

# episodic ltm's are linked by agent, temporal, and semanticly

state[t] = (immediate[t], stm[t], ltm)


# reward module

## discounted signal reconstruction
## Even though immediate prediction, reconstruction, and replay are also performed, this uses a different technique so it may add value to the model
meta_signal_reconstruction_times = range(-T_IFFT_PREV, T_IFFT_FORE+1)
meta_signal_reconstruction = ifft(signal_forier_modes[t], signal_reconstruction_times)
meta_signal_reconstruction_error = meta_signal_forier_modes - meta_signal_reconstruction
meta_signal_reconstruction_discounter = MLP([
  PositionalEncoder(HP('D_META_SIGNAL_RECONSTRUCTOR_PE')),
  Linear(HP('D_META_SIGNAL_RECONSTRUCTOR_FC_1'), activation='relu'),
  Linear(T_IFFT_PREV+T_IFFT_FORE, activation='sigmoid'), # sigmoid to ensure output in (0, 1)
]) # I thought about flipping the gradients to make it want to learn confusing signals, but I'm not sure that's a good idea
meta_signal_reconstruction_discount[t] = meta_signal_reconstruction_discounter(meta_signal_reconstruction_times)
discounted_meta_signal_reconstruction_error = meta_signal_reconstruction_discount[t] * meta_signal_reconstruction_error
total_discounted_meta_signal_reconstruction_error = treesum(discounted_meta_signal_reconstruction_error, weighted=True)


# although fixed/learnable_reward looks like a flat dict, it might actually be hierarchically organized -- perhaps nodes appearing under more than one branch 
fixed_reward[t]: TREE = {
  'discounted_meta_signal_reconstruction_error': -total_discounted_meta_signal_reconstruction_error
  'immediate_prediction': 
}
learnable_reward[t]: Tree = {

}
reward_tree = Tree({ 'fixed_reward': fixed_rewards[t], 
                     'learnable_rewards': learnable_rewards[t] })
w_fixed_reward[t]: TREE.like(fixed_rewards, include_root=True) = operator-supplied parameters
w_learnable_reward[t]: TREE.like(fixed_rewards, include_root=True) = 
reward[t] = treesum(reward, weights={
  'fixed_reward': w_fixed_reward[t],
  'learnable_reward': w_learnable_reward[t],
})
total_fixed_reward = total_reward['fixed_reward']
total_learnable_reward = total_reward['learnable_reward'] 

# TODO: generate different reward signals for different modules, layers, and even neurons

pred_immediate[t] = world_model(state[t])

# TODO: Add Q-function for each reward signal, cumulatively




==========================

Operation modes:

- Collect
  - policy-based action
  - planning-based action (day-dreaming)
  - rainbow action
- Train
  - policy
  - world model
  - reward modules
  - dreaming (night-dreaming)
  - 
- Runtime self-modification










==========================

# Utility functions/primitives

# Node Graph

## Immediate State Processing

## STM Processing

## LTM Processing

## Reward Processing

# Execution Modes

Algorithm specifics for each mode:

- Collect
  - policy-based action
  - planning-based action (day-dreaming)
  - rainbow action
- Train
  - policy
  - world model
  - reward modules
  - dreaming (night-dreaming)
- Runtime self-modification
- Evaluate

Actually, all the modes should be combined into one, and you should just be able to turn on and off specific computations (make ablations to the full run-train-self-modifying algorithm) as easily as turning a light bulb on or off.
- Training has to go back in time. Dreaming has to be disconnected from the world. so really, there need to be separate modes for each of those. However, each of these should be highly paramtrizable/reconfigurable.

Collect (action selection algorithm?, allow self-modification?, etc.)
Train (policy, world model, reward modules, dreaming, etc.)

Gradients are only computed for each time step, but they are pass down from one timestep to the next.

State can be stored inside the class and yet still support stateless execution like so:
```python
instance.do(obs)
Class.do(instance, obs)
```


1. Propagate, compute, compound, and apply gradients on each step

You should be able to dynamically decentralize execution. Since this uses the Python interpreter, that means being able to basically package up the python and send it to a remote machine.

```python

def act(self, obs):
def train(self, traj):

def step(self):

```