import events as e

from .task2_metrics import Task2Metrics


class Task4Metrics(Task2Metrics):

    def record_events(
        self,
        events,
        reward,
        old_game_state=None,
        new_game_state=None,
    ):
        coins_before = self.coins
        super().record_events(events, reward, old_game_state, new_game_state)

        self.record_scores(old_game_state)
        self.record_scores(new_game_state)

        if old_game_state is None:
            return

        step = old_game_state["step"]

        if self.first_elimination_step is None and e.KILLED_OPPONENT in events:
            self.first_elimination_step = step

        if self.fifth_coin_step is None and coins_before < 5 <= self.coins:
            self.fifth_coin_step = step

        if self.death_step is None and e.GOT_KILLED in events:
            self.death_step = step

        if new_game_state is None:
            name, score, _, _ = old_game_state["self"]
            self.scores[name] = (
                score
                + events.count(e.COIN_COLLECTED)
                + 5 * events.count(e.KILLED_OPPONENT)
            )

    def record_scores(self, game_state):
        """Remember the latest visible score for every agent."""
        if game_state is None:
            return

        agent = game_state["self"]
        self.agent_name = agent[0]
        self.scores[agent[0]] = agent[1]
        for name, score, _, _ in game_state.get("others", ()):
            self.scores[name] = score

    def to_dict(self, episode, steps):
        metric = super().to_dict(episode, steps)
        deaths = int(self.killed)
        suicides = int(self.suicide)
        winner = (
            max(self.scores, key=lambda name: (self.scores[name], name))
            if self.scores
            else None
        )

        metric.update(
            {
                "won": self.agent_name is not None and self.agent_name == winner,
                "deaths": deaths,
                "suicides": suicides,
                "kill_death_ratio": self.kills / max(1, deaths),
                "kill_suicide_ratio": self.kills / max(1, suicides),
                "steps_survived": self.death_step or steps,
                "survived_first_50_steps": self.death_step is None
                or self.death_step > 50,
                "survived_first_100_steps": self.death_step is None
                or self.death_step > 100,
            }
        )
        if self.first_elimination_step is not None:
            metric["steps_to_first_elimination"] = self.first_elimination_step
        if self.fifth_coin_step is not None:
            metric["steps_to_five_coins"] = self.fifth_coin_step
        return metric

    def reset(self):
        super().reset()
        self.agent_name = None
        self.scores = {}
        self.death_step = None
        self.first_elimination_step = None
        self.fifth_coin_step = None
