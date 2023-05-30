# TO DO
- Go from 3 channels to 1 (grayscale): The screen images have 3 channels, but our agent only needs 1 channel. So convert the images to grayscale
- Downscale to 84x84 pixels: Images are reasonably large, which makes training slow. Hence resize the image
- Take max of previous 2 frames
- Repeat each action for 4 steps
- Swap channels to first position: Pytorch expects images to have channels first, while the OpenAI Gym returns images with channels last
- Stack 4 most recent frames
- Scale inputs: Divide images by 255

Instantiate our environment and then pas it as an input variable to our custom wrapper

# FEW IMPORTANT POINTS

- In the case of Atari games, the authors of the paper suggested to stack 4 subsequent frames together and use them as the observation at every state. For this reason, the preprocessing stacks four frames together resulting in a final state space size of 84 by 84 by 4.
- Unlike until now we presented a traditional reinforcement learning setup where only one Q-value is produced at a time, the Deep Q-network is designed to produce in a single forward pass a Q-value for every possible action available in the Environmen. This approach of having all Q-values calculated with one pass through the network avoids having to run the network individually for every action and helps to increase speed significantly. 
- Several transformations (as the already introduced the conversion of the frames to grayscale, and scale them down to a square 84 by 84 pixel block) is applied to the Atari platform interaction in order to improve the speed and convergence of the method.

# PSEUDOCODE

```python

class RepeatActionAndMaxFrame # responsible for repeating the actions and handling the max over 2 frames
    derives from: gym.wrapper
    input: environment, repeat # the number of repeat frames is different than the 2 frames over which we take the max to deal with the rendering issue from the Atari Library
    init frame buffer as an array of zeros in shape 2 x the obs space

    function step: # overload the step function for our environment
        input: action
        set total reward to 0
        set done to False
        for i in range repeat
            call the env.step function
                receive obs, reward, done, info
            increment total reward
            insert obs in frame buffer
            if done
                break
        end for
        find the max frame
        return: max frame, total reward, done, info

    function reset:
        input: none

        call env.reset
        reset the frame buffer
        store initial observation in buffer and the zeroth position
        return: initial observation


class PreprocessFrame
    derives from: gym.ObservationWrapper
    input: environment, new shape # new shape that we want to reshape the observation space to
    set shape by swapping channels axis
    set observation space to new shape using gym.spaces.Box (0 to 1.0) # The screen obtained from the emulator is encoded as a tensor of bytes with values from 0 to 255, which is not the best representation for an NN. So, we need to convert the image into floats and rescale the values to the range [0.0…1.0].

    function observation
        input: raw observation
        convert the observation to gray scale # To reduce this complexity, it is performed some minimal processing: convert the frames to grayscale, and scale them down to a square 84 by 84 pixel block.
        resize observation to new shape
        convert observation to numpy array # convert cv2 object to numpy array
        move observations channel axis from position 2 to position 0 # The input shape of the tensor has a color channel as the     
                                                # last dimension, but PyTorch’s convolution layers assume the color channel to be the first dimension. This simple wrapper changes the shape of the observation from HWC (height, width, channel) to the CHW (channel, height, width) format required by PyTorch.
        observation /= 255 # scaling because we set the new shape of observation space between 0 and 1
        return observation

class StackFrames
    derives from: gym.ObservationWrapper
    input: environment, stack size
    init the new obs space (gym.spaces.Box) low & high bounds as repeat of n_steps
    initialize empty frame stack

    reset function
        clear the stack
        reset the environment
        for i in range(stack size)
            append initial observation to stack
        convert stack to numpy array
        reshape stack array to observation space low shape
        return stack

    observation function
        input: observation
        append the observation to the end of the stack
        convert stack to a numpy array
        reshape stack to observation space low shape 
        return the stack of frames

function make_env:
    input: environment name, new shape, stack size
    init env with the base gym.make function
    env := RepeatActionAndMaxFrame
    env := PreprocessFrame
    env := StackFrames

    return: env    
```


# Experience Replay

https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c

The basic idea behind experience replay is to storing past experiences and then using a random subset of these experiences to update the Q-network, rather than using just the single most recent experience. In order to store the Agent’s experiences, we used a data structure called a deque in Python’s built-in collections library. It’s basically a list that you can set a maximum size on so that if you try to append to the list and it is already full, it will remove the first item in the list and add the new item to the end of the list. The experiences themselves are tuples of [observation, action, reward, done flag, next state] to keep the transitions obtained from the environment.

# Target Network

To make training more stable, there is a trick, called target network, by which we keep a copy of our neural network and use it for the Q(s’, a’) value in the Bellman equation.

The predicted Q values of this second Q-network called the target network, are used to backpropagate through and train the main Q-network. It is important to highlight that the target network’s parameters are not trained, but they are periodically synchronized with the parameters of the main Q-network. The idea is that using the target network’s Q values to train the main Q-network will improve the stability of the training.