from typing import Dict, Tuple, Iterable
import torch
from torch import optim
from pathlib import Path
import itertools
import numpy as np
from hac.hac.explorers import Alice, Bob


class ExplorationPolicy:
    def __init__(
        self,
        env,
        layer_number,
        learning_rate: float = 0.001,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ):

        if layer_number == 0:
            action_size = env.action_dim
            state_size = env.state_dim

            action_max, action_min = np.array(env.action_bounds), -1.0 * np.array(env.action_bounds)
            state_max, state_min = np.array([i[1] for i in env.state_bounds]), np.array([i[0] for i in env.state_bounds])
            #state_max, state_min = np.array([float('inf') for i in env.initial_state_space]), np.array([float('-inf') for i in env.initial_state_space])
        else:
            action_size = env.subgoal_dim
            state_size = env.state_dim

            action_max, action_min = np.array([i[1] for i in env.subgoal_bounds]), np.array([i[0] for i in env.subgoal_bounds])
            state_max, state_min = np.array([i[1] for i in env.state_bounds]), np.array([i[0] for i in env.state_bounds])
            #state_max, state_min = np.array([float('inf') for i in env.initial_state_space]), np.array([float('-inf') for i in env.initial_state_space])

        action_min = torch.FloatTensor(action_min).to(device)
        action_max = torch.FloatTensor(action_max).to(device)
        state_min = torch.FloatTensor(state_min).to(device)
        state_max = torch.FloatTensor(state_max).to(device)

        self.alice = Alice(
            state_size=state_size,
            action_size=action_size,
            action_min=action_min,
            action_max=action_max,
        ).to(device)
        self.alice_optimizer = optim.Adam(self.alice.parameters(), lr=learning_rate)
        self.alice_loss = torch.nn.MSELoss()

        self.bob = Bob(
            state_size=state_size,
            action_size=action_size,
            state_min=state_min,
            state_max=state_max,
        ).to(device)
        self.bob_optimizer = optim.Adam(self.bob.parameters(), lr=learning_rate)
        self.bob_loss = torch.nn.MSELoss()

    def get_action(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.alice(state, action).detach()

    def update(
        self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor,
    ) -> None:

        # Get Bob's predictions for next states
        exploratory_actions = self.alice(states, actions)
        pred_next_states = self.bob(states, exploratory_actions)

        # Update Bob. Bob's loss is codified by the notion of "surprise":
        # How wrong his prediction for the next states was vs the actual next states.
        bob_loss = self.bob_loss(pred_next_states, next_states)
        self.bob_optimizer.zero_grad()
        bob_loss.backward()
        self.bob_optimizer.step()

        # # Update Alice. Alice's loss is simply the negative of Bob's (she wants to surprise him).
        alice_loss = -1 * bob_loss.detach().requires_grad_()
        self.alice_optimizer.zero_grad()
        alice_loss.backward()
        self.alice_optimizer.step()

        # print("Exp Actions", exploratory_actions)
        # print("Pred Next States", pred_next_states)
        # print("Next States", next_states)
        # print("Bob Loss", bob_loss.item())

    def save(self, save_location: Path) -> None:
        torch.save(self.alice.state_dict(), save_location / "alice.pth")
        torch.save(self.bob.state_dict(), save_location / "bob.pth")

    def load(self, save_location: Path) -> None:
        self.alice.load_state_dict(torch.load(save_location / "alice.pth"))
        self.bob.load_state_dict(torch.load(save_location / "bob.pth"))