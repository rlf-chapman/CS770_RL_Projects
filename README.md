# CS770_RL_Projects

Group Project submission for CS770 -- Reinforcement Learning

### Currrent progress

 - Video Poker
   - code runs with checkpoints
     - Be sure to update checkpoint location if running elsewhere 
   - multiplay has mostly convereged, seriously diminishing returns (avg reqard is ~210, up from 190 after an aditional 1000 training iterations), single play has converged at a final maximum reward average (~41)
   - I am still imporiving this on my machine, I expect convergence today based on progress.
- Atari Breakout
  - Environment now successfully builds, currently training.
  - Massive performance improvements made by changin hyperparameters and modifying number of environments for each worker
  - workers now also training on GPU simultaneously
