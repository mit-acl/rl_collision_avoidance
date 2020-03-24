
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

### Minimum working example

To start a GA3C training run:
```bash
./example.sh
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