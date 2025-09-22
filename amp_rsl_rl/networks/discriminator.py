# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd
from typing import List, Tuple, Optional, Sequence


class Discriminator(nn.Module):
    """Discriminator for AMP - aligned with TokenHSI implementation.

    This implementation includes all the regularization terms from TokenHSI:
    - Basic BCE loss for expert/policy classification
    - Logit regularization (weight squared penalty)
    - Gradient penalty for training stability
    - Weight decay for discriminator parameters
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: Sequence[int],
        reward_scale: float,
        reward_clamp_epsilon: float = 1e-4,
        device: str | torch.device = "cpu",
        loss_type: str = "BCEWithLogits",
        # TokenHSI alignment parameters
        disc_logit_reg: float = 0.01,
        disc_grad_penalty: float = 10.0,
        disc_weight_decay: float = 0.0,
    ):
        super().__init__()

        self.device = torch.device(device)
        self.input_dim = int(input_dim)
        self.reward_scale = float(reward_scale)
        self.reward_clamp_epsilon = float(reward_clamp_epsilon)

        # TokenHSI alignment parameters
        self.disc_logit_reg = float(disc_logit_reg)
        self.disc_grad_penalty = float(disc_grad_penalty)
        self.disc_weight_decay = float(disc_weight_decay)

        # MLP trunk
        layers: List[nn.Module] = []
        curr_in = self.input_dim
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(curr_in, int(h)))
            layers.append(nn.ReLU())
            curr_in = int(h)
        self.trunk = nn.Sequential(*layers).to(self.device)

        # Final logit head
        if len(hidden_layer_sizes) == 0:
            raise ValueError("hidden_layer_sizes cannot be empty")
        self.linear = nn.Linear(int(hidden_layer_sizes[-1]), 1).to(self.device)

        print(f"Discriminator MLP: {nn.Sequential(self.trunk, self.linear)}")

        self.trunk.train()
        self.linear.train()

        # Loss function
        self.loss_type = (loss_type or "BCEWithLogits").strip()
        if self.loss_type == "BCEWithLogits":
            self.loss_fun: nn.Module = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning discriminator logits"""
        x = x.to(self.device)
        h = self.trunk(x)
        d = self.linear(h)
        return d

    @torch.no_grad()
    def predict_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        normalizer: Optional[object] = None,
    ) -> torch.Tensor:
        """Convert discriminator output to adversarial reward"""
        s = state.to(self.device)
        ns = next_state.to(self.device)

        if normalizer is not None and hasattr(normalizer, "normalize"):
            s = normalizer.normalize(s)
            ns = normalizer.normalize(ns)

        logits = self.forward(torch.cat([s, ns], dim=-1))

        # AMP reward: -log(1 - D(x)), D = sigmoid(logit)
        prob = torch.sigmoid(logits)
        one_minus_p = torch.clamp(1 - prob, min=self.reward_clamp_epsilon)
        reward = -torch.log(one_minus_p)
        reward = self.reward_scale * reward.squeeze(-1)
        return reward

    def get_disc_logit_weights(self) -> torch.Tensor:
        """Get discriminator logit layer weights for regularization - TokenHSI alignment"""
        return torch.flatten(self.linear.weight)

    def get_disc_weights(self) -> List[torch.Tensor]:
        """Get all discriminator weights for regularization - TokenHSI alignment"""
        weights = []
        
        # Collect trunk weights
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                weights.append(torch.flatten(m.weight))
        
        # Add logit layer weights
        weights.append(torch.flatten(self.linear.weight))
        return weights

    def _disc_loss_neg(self, disc_logits: torch.Tensor) -> torch.Tensor:
        """BCE loss for policy samples (target = 0) - TokenHSI alignment"""
        return self.loss_fun(disc_logits, torch.zeros_like(disc_logits, device=self.device))

    def _disc_loss_pos(self, disc_logits: torch.Tensor) -> torch.Tensor:
        """BCE loss for expert samples (target = 1) - TokenHSI alignment"""
        return self.loss_fun(disc_logits, torch.ones_like(disc_logits, device=self.device))

    def compute_loss(
        self,
        policy_d: torch.Tensor,
        expert_d: torch.Tensor,
        sample_amp_expert: Tuple[torch.Tensor, torch.Tensor],
        sample_amp_policy: Tuple[torch.Tensor, torch.Tensor],
        lambda_: float = 10.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute discriminator loss with all TokenHSI regularization terms.
        
        This implementation exactly matches TokenHSI's _disc_loss method:
        1. Basic prediction loss (BCE)
        2. Logit regularization
        3. Gradient penalty
        4. Weight decay
        
        Returns:
            Tuple of (disc_loss, grad_pen_loss) for backward compatibility
        """
        # 1. Basic prediction loss - exactly like TokenHSI
        disc_loss_agent = self._disc_loss_neg(policy_d)
        disc_loss_demo = self._disc_loss_pos(expert_d)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # 2. Logit regularization - exactly like TokenHSI
        if self.disc_logit_reg > 0:
            logit_weights = self.get_disc_logit_weights()
            disc_logit_loss = torch.sum(torch.square(logit_weights))
            disc_loss += self.disc_logit_reg * disc_logit_loss

        # 3. Gradient penalty - compute on expert data like TokenHSI
        grad_pen_loss = torch.tensor(0.0, device=self.device)
        if self.disc_grad_penalty > 0:
            expert_state, expert_next_state = sample_amp_expert
            expert_data = torch.cat([
                expert_state.to(self.device), 
                expert_next_state.to(self.device)
            ], dim=-1)
            
            expert_data = expert_data.detach()
            expert_data.requires_grad_(True)
            
            expert_logits = self.forward(expert_data)
            
            # Compute gradients
            disc_demo_grad = torch.autograd.grad(
                outputs=expert_logits,
                inputs=expert_data,
                grad_outputs=torch.ones_like(expert_logits),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            
            # Gradient penalty like TokenHSI
            disc_demo_grad_norm = torch.sum(torch.square(disc_demo_grad), dim=-1)
            grad_pen_loss = torch.mean(disc_demo_grad_norm)
            disc_loss += self.disc_grad_penalty * grad_pen_loss

        # 4. Weight decay - exactly like TokenHSI
        if self.disc_weight_decay > 0:
            disc_weights = self.get_disc_weights()
            disc_weights_cat = torch.cat(disc_weights, dim=-1)
            disc_weight_decay_loss = torch.sum(torch.square(disc_weights_cat))
            disc_loss += self.disc_weight_decay * disc_weight_decay_loss

        return disc_loss, grad_pen_loss

    def compute_disc_acc(self, disc_agent_logit: torch.Tensor, disc_demo_logit: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute discriminator accuracy - TokenHSI alignment"""
        agent_acc = disc_agent_logit < 0
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc

    # Legacy methods for backward compatibility
    def policy_loss(self, discriminator_output: torch.Tensor) -> torch.Tensor:
        """Legacy method - use _disc_loss_neg instead"""
        return self._disc_loss_neg(discriminator_output)

    def expert_loss(self, discriminator_output: torch.Tensor) -> torch.Tensor:
        """Legacy method - use _disc_loss_pos instead"""
        return self._disc_loss_pos(discriminator_output)

    def compute_grad_pen(
        self,
        expert_states: Tuple[torch.Tensor, torch.Tensor],
        policy_states: Tuple[torch.Tensor, torch.Tensor],
        lambda_: float = 10.0,
    ) -> torch.Tensor:
        """Legacy gradient penalty method - kept for compatibility"""
        expert_s, expert_ns = expert_states
        expert = torch.cat([expert_s.to(self.device), expert_ns.to(self.device)], dim=-1)
        
        expert = expert.detach()
        expert.requires_grad_(True)
        
        scores = self.forward(expert)
        grad = autograd.grad(
            outputs=scores,
            inputs=expert,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gp = grad.norm(2, dim=1).pow(2).mean()
        return lambda_ * gp