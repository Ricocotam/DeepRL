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
