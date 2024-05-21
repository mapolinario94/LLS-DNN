# Table 1
## MNIST
### SmallConv
python main.py --dataset MNIST --model LLS_SmallConv --num-epochs 100 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --waveform cosine --experiment-name Table1
python main.py --dataset MNIST --model LLS_SmallConv --num-epochs 100 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --waveform square --experiment-name Table1
python main.py --dataset MNIST --model LLS_SmallConv --num-epochs 100 --lr 5e-3 --training-mode LLS_Random  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --experiment-name Table1
### VGG8
python main.py --dataset MNIST --model LLS_VGG8 --num-epochs 100 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --waveform cosine --experiment-name Table1
python main.py --dataset MNIST --model LLS_VGG8 --num-epochs 100 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --waveform square --experiment-name Table1
python main.py --dataset MNIST --model LLS_VGG8 --num-epochs 100 --lr 5e-3 --training-mode LLS_Random  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --experiment-name Table1

## FashionMNIST
### SmallConv
python main.py --dataset FashionMNIST --model LLS_SmallConv --num-epochs 100 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --waveform cosine --experiment-name Table1
python main.py --dataset FashionMNIST --model LLS_SmallConv --num-epochs 100 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --waveform square --experiment-name Table1
python main.py --dataset FashionMNIST --model LLS_SmallConv --num-epochs 100 --lr 5e-3 --training-mode LLS_Random  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF3  --experiment-name Table1
### VGG8
python main.py --dataset FashionMNIST --model LLS_VGG8 --num-epochs 100 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --waveform cosine --experiment-name Table1
python main.py --dataset FashionMNIST --model LLS_VGG8 --num-epochs 100 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --waveform square --experiment-name Table1
python main.py --dataset FashionMNIST --model LLS_VGG8 --num-epochs 100 --lr 5e-3 --training-mode LLS_Random  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --experiment-name Table1

## CIFAR10
### SmallConv
python main.py --dataset CIFAR10 --model LLS_SmallConv --num-epochs 100 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --waveform cosine --experiment-name Table1
python main.py --dataset CIFAR10 --model LLS_SmallConv --num-epochs 100 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --waveform square --experiment-name Table1
python main.py --dataset CIFAR10 --model LLS_SmallConv --num-epochs 100 --lr 5e-3 --training-mode LLS_Random  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --experiment-name Table1
### VGG8
python main.py --dataset CIFAR10AUG --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --waveform cosine --experiment-name Table1
python main.py --dataset CIFAR10AUG --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --waveform square --experiment-name Table1
python main.py --dataset CIFAR10AUG --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LLS_Random  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --experiment-name Table1

## IMAGENETTE
### SmallConv
python main.py --dataset IMAGENETTE_BASIC --model LLS_SmallConv --num-epochs 100 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --waveform cosine --experiment-name Table1
python main.py --dataset IMAGENETTE_BASIC --model LLS_SmallConv --num-epochs 100 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --waveform square --experiment-name Table1
python main.py --dataset IMAGENETTE_BASIC --model LLS_SmallConv --num-epochs 100 --lr 5e-3 --training-mode LLS_Random  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --experiment-name Table1
### VGG8
python main.py --dataset IMAGENETTE --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --waveform cosine --experiment-name Table1
python main.py --dataset IMAGENETTE --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --waveform square --experiment-name Table1
python main.py --dataset IMAGENETTE --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LLS_Random  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --experiment-name Table1

# Table 2
## MNIST
python main.py --dataset MNIST --model LLS_SmallConvL --num-epochs 100 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --waveform square --experiment-name Table2
python main.py --dataset MNIST --model LLS_SmallConvL --num-epochs 100 --lr 5e-3 --training-mode BP  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --experiment-name Table2
python main.py --dataset MNIST --model DFA_SmallConvLMNIST --num-epochs 100 --lr 5e-3 --training-mode DFA  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --experiment-name Table2
## CIFAR10
python main.py --dataset CIFAR10 --model LLS_SmallConvL --num-epochs 100 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --waveform square --experiment-name Table2
python main.py --dataset CIFAR10 --model LLS_SmallConvL --num-epochs 100 --lr 5e-3 --training-mode BP  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --experiment-name Table2
python main.py --dataset CIFAR10 --model DFA_SmallConvL --num-epochs 100 --lr 5e-3 --training-mode DFA  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --experiment-name Table2
## CIFAR100
python main.py --dataset CIFAR100 --model LLS_SmallConvL --num-epochs 100 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --waveform square --experiment-name Table2
python main.py --dataset CIFAR100 --model LLS_SmallConvL --num-epochs 100 --lr 5e-3 --training-mode BP  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --experiment-name Table2
python main.py --dataset CIFAR100 --model DFA_SmallConvL --num-epochs 100 --lr 5e-3 --training-mode DFA  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF --experiment-name Table2


# Table 3
## CIFAR10
python main.py --dataset CIFAR10AUG --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode BP  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --experiment-name Table3
python main.py --dataset CIFAR10AUG --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LocalLosses  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF   --experiment-name Table3
python main.py --dataset CIFAR10AUG --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --waveform square --experiment-name Table3
python main.py --dataset CIFAR10AUG --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LLS_M  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --waveform square --experiment-name Table3
python main.py --dataset CIFAR10AUG --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LLS_MxM  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --waveform square --experiment-name Table3
## CIFAR100
python main.py --dataset CIFAR100AUG --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode BP  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF   --experiment-name Table3
python main.py --dataset CIFAR100AUG --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LocalLosses  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF   --experiment-name Table3
python main.py --dataset CIFAR100AUG --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --waveform square --experiment-name Table3
python main.py --dataset CIFAR100AUG --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LLS_M  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --waveform square --experiment-name Table3
python main.py --dataset CIFAR100AUG --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LLS_MxM  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --waveform square --experiment-name Table3
## IMAGENETTE
python main.py --dataset IMAGENETTE --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode BP  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF   --experiment-name Table3
python main.py --dataset IMAGENETTE --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LocalLosses  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF   --experiment-name Table3
python main.py --dataset IMAGENETTE --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --waveform square --experiment-name Table3
python main.py --dataset IMAGENETTE --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LLS_M  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --waveform square --experiment-name Table3
python main.py --dataset IMAGENETTE --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LLS_MxM  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --waveform square --experiment-name Table3
## TinyIMAGENET
python main.py --dataset TinyIMAGENET --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode BP  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF   --experiment-name Table3
python main.py --dataset TinyIMAGENET --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LocalLosses  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF   --experiment-name Table3
python main.py --dataset TinyIMAGENET --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --waveform square --experiment-name Table3
python main.py --dataset TinyIMAGENET --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LLS_M  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --waveform square --experiment-name Table3
python main.py --dataset TinyIMAGENET --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LLS_MxM  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --waveform square --experiment-name Table3
## VWW
python main.py --dataset VWW --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode BP  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF   --experiment-name Table3
python main.py --dataset VWW --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LocalLosses  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF   --experiment-name Table3
python main.py --dataset VWW --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LLS  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --waveform square --experiment-name Table3
python main.py --dataset VWW --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LLS_M  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --waveform square --experiment-name Table3
python main.py --dataset VWW --model LLS_VGG8 --num-epochs 300 --lr 5e-3 --training-mode LLS_MxM  --batch-size 128 --test-batch-size 128 --optimizer AdamWSF  --waveform square --experiment-name Table3


