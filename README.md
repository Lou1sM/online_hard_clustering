Code for the paper [Hard Regularization to Prevent Deep Online Clustering Collapse without Data Augmentation](https://ojs.aaai.org/index.php/AAAI/article/view/29340).

# Installation
Create a conda environment (tested with python 3.12), and run the script that installs the needed libraries: 

```. install_requirements.sh```. 

# Run
The experiments in the paper can be replicated with the command 

```python train.py -d {dataset-name} --n_epochs 10```

The dataset names are 'c10', 'c100', 'fashmnist', 'stl' and 'realdisp', meaning Cifar10, Cifar100, Fashion MNIST, STL and RealDisp, respectively.
The comparison models can be run by adding the flags '--var', '--ent', '--sinkhorn'' or '--ckm'', which correspond to the names used for these models in the paper.

# Citation
If you use or refer to this in your work, please cite
```
  title={Hard Regularization to Prevent Deep Online Clustering Collapse without Data Augmentation},
  author={Mahon, Louis and Lukasiewicz, Thomas},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={13},
  pages={14281--14288},
  year={2024}
}
```

Any questions or problems with the code, you can contact lmahon@ed.ec.uk.
