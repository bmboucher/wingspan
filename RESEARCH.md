This document outlines a series of research projects we want to use our RL model for Wingspan on.

# Related to Model Training

## Project: General architecture exploration
For each submodel in our overall architecture (setup, card encoding, state encoding, choice encoding, value head, each decision head), let's define a range of possible numbers and sizes of hidden layers. There's a lot of sample space to explore here, so I think to keep it simple we can start with a "lite" version (1 small hidden layer) and a "heavy" version (3 larger hidden layers) for every submodel - we want to run a series of training tests where we pin every submodel to its "lite" version except one (which we sweep across all the submodels) which uses "heavy".
Metrics to consider: Runtime/throughput (games/sec), time to converge, relative performance (play them against each other)
Main questions:
1. What is the optimal size/shape for each of the submodels?
2. Does this answer vary between the different decision heads enough that we need to configure each separately for most runs?

# Related to Model Function

## Project: 

# Using the Model for Card Valuation

## Project: Perturbation testing
I want to walk through some games manually in detail, and try to get a sense of the model's "thought process". The raw scores give me some information but they don't tell me the "why". One way to get at it is to adjust the model inputs slightly, recalculate, and determine how much the policy output changes.