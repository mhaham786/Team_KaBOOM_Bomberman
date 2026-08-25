# PPO Agent

Train Coin Heaven (Scene 1) from this directory:

```bash
make train-scene-1
```

Plot its metrics (after training):

```bash
make EXPERIMENT=coin_heaven
```

Edit `config.py` to set `EXPERIMENT_NAME`, PPO values, features, and rewards.
New training is the default: use a new experiment name. Set
`RESUME_TRAINING = True` only to continue an existing experiment.

## Future training improvements

- Generalized Advantage Estimation (GAE)
- Minibatch PPO updates
- CNN model for the board maps
