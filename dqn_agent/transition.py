import numpy as np

from ..common.features import *
from ..common.helpers import action_mask
from ..common.rewards import *
from . import config


class Transition:
    """Manage the newest environment transition until its outcome is known."""

    def __init__(self, agent):
        self.agent = agent
        self.clear()

    def record(self, old_game_state, action, new_game_state, events):
        """Store the previous pending step and hold the newest one."""
        if self.old_game_state is not None:
            self.store(
                self.old_game_state,
                self.action,
                self.new_game_state,
                self.events,
            )

        self.old_game_state = old_game_state
        self.action = action
        self.new_game_state = new_game_state
        self.events = list(events)

    def finish(self, last_game_state, last_action, events):
        """Store the final action once and clear the pending step."""
        if self.old_game_state is not None and not self.matches_final(
            last_game_state,
            last_action,
        ):
            self.store(
                self.old_game_state,
                self.action,
                self.new_game_state,
                self.events,
            )

        self.store(last_game_state, last_action, None, events)
        self.clear()

    def matches_final(self, last_game_state, last_action):
        """Return whether the pending and final callbacks describe one action."""
        return (
            self.old_game_state["round"] == last_game_state.get("round")
            and self.old_game_state["step"] == last_game_state.get("step")
            and self.action == last_action
        )

    def store(self, old_game_state, action, new_game_state, events):
        """Add one completed transition to replay and update training state."""
        agent = self.agent
        done = new_game_state is None
        events = list(events)
        state = advanced_features_oc31(old_game_state)
        next_state = advanced_features_oc31(new_game_state)
        next_mask = (
            np.zeros(len(config.ACTIONS), dtype=np.bool_)
            if done
            else action_mask(new_game_state)
        )
        event_reward = classic_peace_rewards_dqn_improved(events)
        shaping_reward = advanced_reward_shaping_dqn(
            action,
            old_game_state,
            new_game_state,
            events,
            config.GAMMA,
        )
        reward = event_reward + shaping_reward

        agent.replay_buffer.push(
            state,
            config.ACTIONS.index(action),
            reward,
            next_state,
            done,
            next_mask,
        )
        agent.total_transitions += 1
        agent.epsilon = self.epsilon_for(agent.total_transitions)
        agent.round_steps += 1
        agent.round_reward += reward
        agent.round_event_reward += event_reward
        agent.round_shaping_reward += shaping_reward
        agent.metrics.record_events(
            events,
            event_reward,
            old_game_state,
            new_game_state,
        )

        loss = agent.trainer.update(
            agent.replay_buffer,
            agent.total_transitions,
        )
        if loss is not None:
            agent.round_losses.append(loss)

    def clear(self):
        """Clear the pending transition values."""
        self.old_game_state = None
        self.action = None
        self.new_game_state = None
        self.events = None

    @staticmethod
    def epsilon_for(total_transitions):
        """Return epsilon for a transition count using linear decay."""
        fraction = min(
            max(total_transitions, 0) / config.EPSILON_DECAY_STEPS,
            1.0,
        )
        return config.EPSILON_START + fraction * (
            config.EPSILON_END - config.EPSILON_START
        )
