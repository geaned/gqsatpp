# Graph-Q-SAT++

Project structure:
- `gqsat_model_minisat` directory contains the proposed implementation of MiniSat with functionality to consult a trained branching model via C++ PyTorch API (LibTorch)
- `gqsat_custom_trainer` directory contains the Graph-Q-SAT code used for training the branching model (all the important changes can be found in the `gqsat/models.py` file, the original file is located at `gqsat/models_.py`)
