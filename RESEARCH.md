This document outlines a series of research projects we want to use our RL model for Wingspan on.

# Related to Model Training

## Project: General architecture exploration
For each submodel in our overall architecture (setup, card encoding, state encoding, choice encoding, value head, each decision head), let's define a range of possible numbers and sizes of hidden layers. There's a lot of sample space to explore here, so I think to keep it simple we can start with a "lite" version (1 small hidden layer) and a "heavy" version (3 larger hidden layers) for every submodel - we want to run a series of training tests where we pin every submodel to its "lite" version except one (which we sweep across all the submodels) which uses "heavy".
Metrics to consider: Runtime/throughput (games/sec), time to "converge" (based on some definition), relative performance (play them against each other)
Main questions:
1. What is the optimal size/shape for each of the submodels?
2. Does this answer vary between the different decision heads enough that we need to configure each separately for most runs?

# Related to Model Function

## Project: Does the main action model know "which" card it wants to play?
We've split out the choice of main action (play bird, gain food, lay eggs, or draw cards) from the secondary choice of which bird to play. This made sense from a model construction perspective, but I'm curious if the model deciding to play a bird is also calculating the relative value of doing that based on which cards will actually be available to play in the next choice, or if it is just making a high-level call on building the board vs running the engine.
One way to test this is to look at a sample of main action decision points where we had >1 birds available to play, and to look at how the model output changes if we remove each of those birds from the hand. If the score for "play a bird" goes down when we remove one bird, but not when we remove another, that indicates that the main action model is weighting that bird heavily in the decision. We should be able to do a basic statistical analysis to determine how often this happens, and whether the "chosen" bird correlates well with the actual selection from the "play which bird" model.

## Project: Does the model learn not to add eggs when it can't?
One action the game engine will allow that does literally nothing is going to the Grassland when there are no brown power birds in it and no available egg capacity. This option will always be presented to the model for scoring, but we would hope that it learns not to do it from observation. We can test that by running a large number of game and capturing every data point that matches that description (the model is offered a choice to gain no eggs + activate no birds) and seeing what rate the model chooses to lay eggs in this scenario. Our expectation is zero but we should also compare it to the average across all decision points to get a sense of how often this matters in simulated games.

# Using the Model for Card Valuation / Game Analysis

## Project: Setup card stats
There are ~484 billion possible opening hands of 5 bird cards and 2 bonus cards, so we obviously can't run them all through the model. But we can run a very large sample (a few million) and then generate the full list of 504 setup selection options for each, and score all of those in the model to determine what our policy would be for each hand.
Main questions:
1. Which cards in the set have the highest probability of being selected if they are in the opening hand?
2. Which cards in the set have the most impact on the game's eventual outcome?
3. What does our "ideal" opening hand look like?

## Project: General card stats
This one is pretty simple - we just want to lock a model down and play as many games as fast as we can to get a large sample size, capturing metrics we can analyze. Of particular interest:
* The Games In-Hand (GIH) win rate - how often did we win when we drew a given card or had it in our opener?
* The Games On-Board (GOB) win rate - how often did we win when we had a given card on our board? How does that breakdown per slot (i.e. are certain birds much better in certain locations than others)? 

## Project: Are there "low-scoring" and "high-scoring" games?
I often feel when I play in person that some games are "good" for both players and some are "bad" for both players - obviously both players share in the draws off the deck, but there isn't really enough interaction between players to explain this. I suppose it could also be the end-of-round goals, since those sometimes line up well and sometimes don't, so optimizing for them may mean sub-optimal plays in other places. At any rate, the main question is: is there any correlation between P0's score and P1's score? Simple enough statistical metric, we should probably just start including it in the standard analysis for any run.