import torch
import torch.nn as nn


class Alice(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        action_min: torch.tensor,
        action_max: torch.tensor,
        hidden_layer_size: int = 64,
    ):
        super(Alice, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.actor = nn.Sequential(
            # Current state and goal state are combined input to the first layer
            nn.Linear(self.state_size + self.action_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, self.action_size),
        )

        assert (
            action_min.size(dim=-1) == self.action_size
        ), f"Action minima with size {action_min.size()} does not match expected size {self.action_size} in relevant dimension!"
        assert (
            action_max.size(dim=-1) == self.action_size
        ), f"Action maxima with size {action_max.size()} does not match expected size {self.action_size} in relevant dimension!"

        # Parameters to linearly transform output action range from
        # tanh's limited output range of [-1, 1] to environment's actual
        # action range [action_min, action_max]
        self.action_mean = (action_max + action_min) / 2
        self.action_range = (action_max - action_min) / 2

    def forward(self, state: torch.tensor, action: torch.tensor) -> torch.tensor:
        # assert (
        #     state.size(dim=-1) == self.state_size
        # ), f"State with size {state.size()} does not match expected size {self.state_size} in relevant dimension!"
        # assert (
        #     action.size(dim=-1) == self.action_size
        # ), f"Action with size {action.size()} does not match expected size {self.action_size} in relevant dimension!"

        # Concatenate state and action for input
        combined_state_action = torch.cat((state, action), dim=-1)
        action_normalized = self.actor(combined_state_action)

        # Scale the net's [-1, 1] action output to appropriate action range
        action_unnormalized = action_normalized * self.action_range + self.action_mean

        # print("Ac", action_unnormalized)

        return action_unnormalized


class Bob(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        state_min: torch.tensor,
        state_max: torch.tensor,
        hidden_layer_size: int = 64,
    ):
        super(Bob, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        # Simple sequential network
        self.model = nn.Sequential(
            # Current state and action state are combined input to the first layer
            nn.Linear(self.state_size + self.action_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, self.state_size),
        )

        # assert (
        #     state_min.size(dim=-1) == self.state_size
        # ), f"State minima with size {state_min.size()} does not match expected size {self.state_size} in relevant dimension!"
        # assert (
        #     state_max.size(dim=-1) == self.state_size
        # ), f"State maxima with size {state_max.size()} does not match expected size {self.state_size} in relevant dimension!"

        # Parameters to linearly transform output state range from
        # tanh's limited output range of [-1, 1] to environment's actual
        # state range [state_min, state_max]
        self.state_mean = (state_max + state_min) / 2
        self.state_range = (state_max - state_min) / 2

    def forward(self, state: torch.tensor, action: torch.tensor) -> torch.tensor:
        assert (
            state.size(dim=-1) == self.state_size
        ), f"State with size {state.size()} does not match expected size {self.state_size} in relevant dimension!"
        assert (
            action.size(dim=-1) == self.action_size
        ), f"Action with size {action.size()} does not match expected size {self.action_size} in relevant dimension!"

        # Concatenate state and goal for UVFA input
        combined_state_action = torch.cat((state, action), dim=-1)
        next_state_normalized = self.model(combined_state_action)

        # Scale the net's [-1, 1] next_state output to appropriate next_state range
        next_state_unnormalized = (
            next_state_normalized * self.state_range + self.state_mean
        )

        # print("Next", next_state_unnormalized)

        return next_state_unnormalized