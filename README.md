# pytorch-SelectiveNet

This is a pytorch implementation of the paper titled SelectiveNet: A Deep Neural Network with an Integrated Reject Option [Geifman and El-Yaniv, ICML2019].

## Requirements

You will need the following to run the codes:
- Python 3.5+
- Pytorch 1.4+
- CUDA
- wandb
- click

I run the code with Pytorch 1.4.0, CUDA 9.2

### Training
Use `scripts/train.py` to train the network. Example usage:
```bash
# Example usage
cd scripts
python train.py --dataset cifar10 --coverage 0.7 
```

### Testing
Use `scripts/test.py` to test the network. Example usage:
```bash
# Example usage (test single weight)
cd scripts
python test.py --dataset cifar10 --exp_id ${id_of_training_experminet} --weight ${name_of_saved_model}--coverage 0.7

# Example usage (test multiple weights)
cd scripts
python test.py --dataset cifar10 --exp_id 2fkl0ib7 --coverage 0.7
```

## References

- [Yonatan Geifman and Ran El-Yaniv. "SelectiveNet: A Deep Neural Network with an Integrated Reject Option.", in ICML, 2019.][1]
   
- [Original implementation in Keras][2]

[1]: https://arxiv.org/abs/1901.09192
[2]: https://github.com/geifmany/selectivenet
