# [Goal-Conditioned Reinforcement Learning with Imagined Subgoals](https://www.di.ens.fr/willow/research/ris/)

## Environments

The multiworld environments are taken from https://github.com/vitchyr/multiworld/tree/leap

## Ant Navigation

To train RIS on the U-shaped ant maze environment, please run:
```
python train_ant.py --env_name AntU
```

Use this table to run RIS on other ant maze navigation tasks:

| Environment                | --env_name |  
| -------------------------- |:----------:| 
| U-shaped ant maze (default)| AntU       | 
| S-shaped ant maze          | AntFb      | 
| $\Pi$-shaped ant maze      | AntMaze    |
| $\omega$-shaped ant maze   | AntFg      |

## Vision-Based Robotic Manipulation

To train RIS on the vision-based robotic manipulation environment, please run:
```
python train_sawyer.py 
```

