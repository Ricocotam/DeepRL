# DeepRL
Any deep reinforcement learning algorithm I coded. Everything is using PyTorch


## Available algorithms
- DQN
- REINFORCE


## A little framework
I decided to build all the algorithms around a little framework I came with. It is structured as follow

- Model : A model, it can be neural net or anything else
- Policy : Classic algorithm to select an action from the output of the model
- Buffer : Store past experiences so you can replay
- Agent : And agent that takes actions according to Policy applied on Model and sampling experiences from Buffer

The reason I made this framework is to avoid rewriting common code.
