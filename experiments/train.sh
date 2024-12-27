python train.py --dataset.name=cifar10 --model.name=resnet18 --train.max_epochs=90 --train.lr=1e-6 --train.batch_size=32 
python train.py --dataset.name=cifar10 --model.name=resnet18 --train.max_epochs=90 --train.lr=1e-6 --train.batch_size=32
python train_without_forgetting.py --dataset.name=cifar10 --model.name=resnet18 --train.max_epochs=90 --train.lr=1e-6 --train.batch_size=32 --forgetting_set_size=3 --forgetting_set=[1,4,5]
python train_without_forgetting.py --dataset.name=cifar10 --model.name=resnet18 --train.max_epochs=90 --train.lr=1e-6 --train.batch_size=32 --forgetting_set_size=3 --forgetting_set=[3,4,5]
python train_without_forgetting.py --dataset.name=cifar10 --model.name=resnet18 --train.max_epochs=90 --train.lr=1e-6 --train.batch_size=32 --forgetting_set_size=3 --forgetting_set=[0,6,8]
python train_without_forgetting.py --dataset.name=cifar10 --model.name=resnet18 --train.max_epochs=90 --train.lr=1e-6 --train.batch_size=32 --forgetting_set_size=2 --forgetting_set=[1,8]
python train_without_forgetting.py --dataset.name=cifar10 --model.name=resnet18 --train.max_epochs=90 --train.lr=1e-6 --train.batch_size=32 --forgetting_set_size=2 --forgetting_set=[2,4]
python train_without_forgetting.py --dataset.name=cifar10 --model.name=resnet18 --train.max_epochs=90 --train.lr=1e-6 --train.batch_size=32 --forgetting_set_size=2 --forgetting_set=[6,9]
python train_without_forgetting.py --dataset.name=cifar10 --model.name=resnet18 --train.max_epochs=90 --train.lr=1e-6 --train.batch_size=32 --forgetting_set_size=1 --forgetting_set=[3]
python train_without_forgetting.py --dataset.name=cifar10 --model.name=resnet18 --train.max_epochs=90 --train.lr=1e-6 --train.batch_size=32 --forgetting_set_size=1 --forgetting_set=[7]
python train_without_forgetting.py --dataset.name=cifar10 --model.name=resnet18 --train.max_epochs=90 --train.lr=1e-6 --train.batch_size=32 --forgetting_set_size=1 --forgetting_set=[9]
