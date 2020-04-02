
This is the training code for:

M. Everett, Y. Chen, and J. P. How, "Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning", IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2018
*  Paper: https://arxiv.org/abs/1805.01956
*  Video: https://www.youtube.com/watch?v=XHoXkWLhwYQ


### Install

Grab the code from github, initialize submodules, install dependencies and src code
```bash
# Clone either through SSH or HTTPS
# SSH
git clone --recursive git@gitlab.com:mit-acl/ford_ugvs/planning_algorithms/cadrl/rl_collision_avoidance.git
# HTTPS
git clone --recursive https://gitlab.com/mit-acl/ford_ugvs/planning_algorithms/cadrl/rl_collision_avoidance.git

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

### To run experiments on AWS
Start a bunch (e.g., 5) of AWS instances -- I used `c5.2xlarge` because they have 8vCPUs and 16GB RAM (somewhat like my desktop?).

Add the IP addresses into `ga3c_cadrl_aws.sh`.
```bash
./ga3c_cadrl_aws.sh panes
# C-a :setw synchronize-panes -- will let you enter the same command in each instance
```

Then you can follow the install & train instructions just like normal. When training, it will prompt you for a wandb login (can paste in the authorization code from app.wandb.ai/authorize).

### Observed Issues
If on OSX, when running the `./train.sh` script, you see:
```bash
objc[39391]: +[__NSCFConstantString initialize] may have been in progress in another thread when fork() was called.
objc[39391]: +[__NSCFConstantString initialize] may have been in progress in another thread when fork() was called. We cannot safely call it or ignore it in the fork() child process. Crashing instead. Set a breakpoint on objc_initializeAfterForkError to debug.
```
just add this ENV_VAR: `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`.

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