# QuClassi: A Hybrid Deep Neural Network Architecture based on Quantum State Fidelity 
## MLSys 2022 Publication

QuClassi is a Quantum Deep Neural Network architecture for classification, based on quantum state fidelity 

## Usage

To use QuClassi, install the requirements by using 
```bash
pip install -r requirements.txt
```
Within main.py, there is a subsampling section
```python
SUBSAMPLE = 1000
````
This is to be edited according to computational constraints. More data results in slower training speeds, and hence subsamples are used for quicker evaluation.

From here, to run the system, run the command
```bash
python main.py
```
Subsample sets can be edited by editting the training labels and training datasets accordingly. 

[MLSys Link - Pending](127.0.0.0)
[Arxiv Link](https://arxiv.org/abs/2103.11307)]
