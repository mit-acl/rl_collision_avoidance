
This is the training code for:

M. Everett, Y. Chen, and J. P. How, "Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning", IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018
*  Paper: https://arxiv.org/abs/1805.01956
*  Video: https://www.youtube.com/watch?v=XHoXkWLhwYQ


### Install

Grab the code from github, initialize submodules, install dependencies and src code
```bash
git clone --recursive git@gitlab.com:mit-acl/ford_ugvs/planning_algorithms/cadrl/rl_collision_avoidance.git
cd rl_collision_avoidance
./install.sh
```

### Train RL (starting with a network initialized through supervised learning on CADRL decisions)

To start a GA3C training run (it should get approx -0.05-0.05 rolling reward to start):
```bash
./train.sh TrainPhase1
```

To load that checkpoint and continue phase 2 of training, update the `LOAD_FROM_WANDB_RUN_ID` path in `Config.py` and do:
```bash
./train.sh TrainPhase2
```

### If you find this code useful, please consider citing:

```
@inproceedings{Everett18_IROS,
  address = {Madrid, Spain},
  author = {Everett, Michael and Chen, Yu Fan and How, Jonathan P.},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  date-modified = {2018-10-03 06:18:08 -0400},
  month = sep,
  title = {Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning},
  year = {2018},
  url = {https://arxiv.org/pdf/1805.01956.pdf},
  bdsk-url-1 = {https://arxiv.org/pdf/1805.01956.pdf}
}
```