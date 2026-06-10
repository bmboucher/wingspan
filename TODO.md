# Issues with the detailed game log from `wingspan play --log`

1. Saw this:
```
[P0] @ Burrowing Owl - "Roll all dice not in birdfeeder. If any are rodent, cache 1 rodent from the supply on this bird."
[P0: AcceptExchangeDecision | 2 choices | greedy]
1. roll dice
    66.496%  ( -2.22)
2. skip
    33.504%  ( -2.90)
[P0 chose: roll dice (66.496%)]
```
This is an always beneficial power - there's no cost associated with rolling dice NOT in the birdfeeder, and caching from the supply is always free. We don't want to overload the skip_optional head with this.

2. During the "CHOOSING BIRDS" part of setup, let's also include the two bonus cards. Display these the way are in the start-of-turn display so we can see the full text on the first line, but on the second indented line let's include matching counts from the hand and tray.

3. In the "CHOOSING BIRDS" section, we're still printing "foods:[none] bonus:(none)" when we have the initial food/bonus selection split out; we need to drop this so we're only showing the list of kept birds for each option.

4. Let's split the log that `wingspan play --log {FILE}.log` into a `{FILE}_p0.log` and `{FILE}_p1.log` by default. Add an option `--collate` to show the two players turns correctly interleaved.

5. For each decision, let's indent the numbered list of choices a bit; the percentages/scores are in a good location but I want to see `[P#: BlaBlaDecision ... ]` on one indentation level and then the next thing on that level should be `[P# chose: ...]`

6. Let's make the log lines more uniform - right now some of them look like `[P# ...]` and others look like `[P#] ...`, I prefer the latter exclusively.

7. For each XYZDecision model in the log, let's also print the name of the decision head (skip_optional etc) that handles it.

8. When we have decisions made by the other player during one player's turn (e.g. Anna's Hummingbird), it looks like we're still printing that player in the log as the one that "chooses". I'm assuming that these options are presented to the model with the is_self flag set to 0 *as well as POV rotation*, please confirm that. Make sure the log reflects the correct player making the decision, i.e. if is_self=0 and it's P0's turn we should print `[P1]` lines.

# Training feature - bootstrap against pre-trained opponent

Right now we're bootstrapping against a random opponent until we reach some "graduation" threshold, then switching over to self-play. I'd like the option to train against a pre-trained model as opponent instead (graduating to self-play if we reach the win rate threshold). We shouldn't require that both player agents have the same architecture, we should be able to load parameters saved from any previous run per the backwards compatibility policy and run it as our "bootstrap" model.
Also the configurator UI will need to be updated to reflect this - the most likely use pattern is that we run training for some period of time, while we make concurrent changes to the model that may bump the model version number, and then we want to restart training with the most recent model but using our trained output from the first run as the opponent during training.

# Modeling - use one-hots for round number and cube count

Self explanatory from the title, I think using a raw value for round number is really bad and using one for number of cubes is probably bad. Let's update the model to use one-hots for these instead, with a minor version bump for backwards compatibility.

# General investigation - these don't require action, just write detailed reports in the docs folder for each

1. I'm noticing after long training runs that the average points gained from cached food is small compared to other sources. This could make sense, we don't have bonus cards in the base set that care about cached food etc, but I want to double-check how every caching bird is handled. Are we correctly exposing the information about caching to the model?

2. Would it make sense to use a self-attention layer for "board encoding"? What I'm thinking is we have two boards, each with 15 stripes that consist of the single-card embedding vector + state values (e.g. eggs). Could we add a layer that represents each slot on the board "paying attention" to all the other slots? i.e. the input is `15*stripe` and the output is the same size, but within the attention layer we have 15 "units". Explain to me how if this would work, how it would work, and whether or not its a good idea.

3. How are the rewards calculated for training the critic? Is it just assigning the ending point value back to every previous step, or do we calculate points before and after each decision point and assign rewards that way?