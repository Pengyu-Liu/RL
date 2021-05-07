#Instruction about how to run the codes.
#This folder contains three python files: ddpg.py for Deep Deterministic Policy Gradient; ppo.py for Proximal optimization; A3C.py for Asynchronous Advantage Actor-Critic written with tensorflow 1, and A3C_torch(just for comparison).py written with pytorch just for comparison experiment.

# If you want to repeat all the experiments we showed in our report, read the code and change values of hyper-parameters. We have given hints in comments.


#run Proximal optimization (with tensorflow 2)
python ppo.py

#run Deep Deterministic Policy Gradient (with pytorch)
python ddpg.py

#run Asynchronous Advantage Actor-Critic (with tensorflow 1)
python A3C.py
