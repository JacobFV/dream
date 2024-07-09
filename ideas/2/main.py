import tensacode


class Agent:
    def __init__(self):
        self.tc = tensacode.Engine()

    def step(self):

        # observation
        ## apply input encoders to generate normalized signals
        ## compare against prediction errors
        ## hierarchical modality fusion
        ## unsupervised cost objectives (L2, stddev, etc.)

        # memory retrieval
        ## differentiable short term memory retrieval
        ## differentiable long term memory retrieval
        ## non-differentiable long term memory retrieval
        ## cross agent attention

        # thinking
        ## integrate sensory, memory, and prev action into current state
        ## sample qRNG,biased by the nondeterministic factor
        ## update bottleneck graph

        # memory updating
        ## update memory graph
        ## randomly offload distant unused nodes from memory
        ## update modality specific memory systems

        # action
        # predict next state
        # predict rollout

        # if training:
        ## discounted rollout accuracy penalty (use the appropriate loss fn for each signal)
        ## apply gradients with the new optimizer

        # if evolving:
        ## evolutionary prunning
        ## evolutionary rewinding
        ## evolutionary merge (merge in specializations)
        ## evolutionary spawn

        pass
