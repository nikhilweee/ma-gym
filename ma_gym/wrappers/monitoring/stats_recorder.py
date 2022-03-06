from gym.wrappers.monitoring.stats_recorder import StatsRecorder as SR


class StatsRecorder(SR):
    def __init__(self, *args, **kwargs):
        self.info = {}
        self.infos = {}
        super().__init__(*args, **kwargs)

    def after_step(self, observation, reward, done, info):
        self.info.update(info)
        super().after_step(observation, sum(reward), all(done), info)

    def save_complete(self):
        if not self.infos:
            for key in self.info:
                self.infos[key] = [self.info[key]]
        else:
            for key in self.infos:
                self.infos[key].append(self.info[key])
        super().save_complete()

    def flush(self):
        self.infos['total_wins'] = [0, 0]
        self.infos['average_score'] = [0, 0]
        for reward in self.infos['rewards']:
            self.infos['average_score'][0] += reward[0]
            self.infos['average_score'][1] += reward[1]
            if reward[0] > reward[1]:
                self.infos['total_wins'][0] += 1
            if reward[1] > reward[0]:
                self.infos['total_wins'][1] += 1
        self.infos['average_score'][0] /= len(self.infos['rewards'])
        self.infos['average_score'][1] /= len(self.infos['rewards'])
        self.episode_rewards = self.infos
        super().flush()