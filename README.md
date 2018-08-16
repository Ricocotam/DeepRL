# DeepRL
Any deep reinforcement learning algorithm I coded. Everything is using PyTorch


## Available algorithms
- DQN
- REINFORCE


## A little framework
I decided to build all the algorithms around a little framework I came with. It is structured as follow

- Model : A model, designed to be a neural network
- Policy : Classic algorithm to select an action from the output of the model
- Buffer : Store past experiences so you can replay
- Agent : And agent that takes actions according to Policy applied on Model and sampling experiences from Buffer

The reason I made this framework is to avoid rewriting common code.

#### Model
##### Attributes
- loss_function : The loss function used to learn
- optim : The optimiser used to learn
- gamma : The gamma parameter from MDP definition

##### Methods
- \__init__
  Initialise the model
  - Parameters
    - gamma : float
      Gamma parameter from MDP definition
    - optim : torch.optim.Optimizer
      The optimizer used to learn the model
    - loss_function : torch.nn.\_Loss
      The loss function used to optimize.
    - device : torch.device, optionnal
      The device to store data on

  - Note
    If you wish to initialise anything before passing it to the model, use itertools.partial.

- \__call__, abstract
  Predict the next action (could be values, probabilities...)
  - Parameters
    - state : custom
      The state of the environment

- learn, abstract
  Learn from given experiences
  - Parameters
    - sample : list of experience
      A sample from which the model should learn

- update, abstract
  Update model parameters. Useful, for example, on double learning

#### Policy
##### Methods
- \__call__, abstract
  Select action from model prediction on state
  - Parameters
    - state : custom
      The state of the environment
    - model : Model
      The model used to predict
- update, abstract
  Update the policy. Useful, for example, on epsilon decay policy


Currently implemented policies :
* Greedy
* EpsGreedy : classical epsilon-greedy. Takes epsilon and action size as parameters
* EpsDecay : Extends epsilon greedy using a decay. Takes eps_start, eps_min, eps_decay and action size as parameters. Call update to change epsilon : eps = max(eps_min, eps * eps_decay)
* SoftmaxPolicy : Applies softmax to the model output and sample from it

#### ReplayBuffer
##### Attributes
- memory : collections.deque
  A deque to store the experiences
- batch_size : int
  Size of a batch during training
- device : torch.device, optionnal
  The device to send data on

##### Methods
- \__init__
  Initialise a buffer
  - Parameters
    - buffer size : int
      Size of the memory
    - batch_size : int
      Size of a batch during training
    - device : torch.device, optionnal
      The device to send data on

- add, abstract
  Add a new experience (can be a step or a whole episode)
  - Parameters
    - step : custom
      A step of the alg (usually something like state, action, reward)
- sample, abstract
  Sample batch_size experiences from memory
- can_sample, abstract
  Returns True if the buffer can sample data, False either.
  Default is True if the number of experience in memory is over the batch size
- \__len__
  Length of the buffer. Default is len(memory)


Currently implemented buffers :
- SoloBuffer : Can only store one experience. Useful when you don't need buffer (policy gradients for example)
- QBuffer : Buffer designed for Q-learning. Experiences are stored as a named tuple with fields state, action, reward, next_state, done. One step is an ordered iterable of those values.
- CompleteBuffer : Just like QBuffer but with more data : state, action, reward, next_state, next_action, done

#### Agent
##### Attributes
- model : Model
  The model the agent will use to predict actions
- buffer : Buffer, optionnal
  A replay buffer, default is SoloBuffe
- learning_strategy : Policy
  The policy to follow during training
- playing_strategy : Policy
  The policy to follow when not training
- update_every : int
  Update the model every "update_every" call on step.
- learn_every : int
  Learn the model every "learn_every" call on step.
- learning : bool
  True if the agent is learning. Changing it changed the policy used.

#### Methods
- \__init__
  - model : Model
    The model the agent will use to predict actions
  - learning_strategy : Policy
    The policy to follow during training
  - policy_playing : Policy, optionnal
    The policy to follow when not training. Default Greedy
  - buffer : Buffer, optionnal
    A replay buffer, default is SoloBuffer
  - update_every : int, optionnal
    Update the model every "update_every" call on step. Default 1
  - learn_every : int, optionnal
    Learn the model every "learn_every" call on step. Default 1
- act
  Get the action from the state.
  - Parameters
    - state, custom
      The state from which the agent have to take an action.
  - Returns
    The action the agent picked.
- step
  Do a step for the agent. Memorize and learn.
  Append the given experience to the buffer and sample from the buffer so it can learn and update if needed (@see learn_every and update_every)
  - Parameters
    - experience : custom
      The last experience you had. It is added to the buffer
- learning
  Change learning attributes to True. It enables the training policy
- playing
  Change learning attribute to False. It enables the playing strategy
