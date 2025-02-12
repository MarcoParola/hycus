python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.scrub_steps=7 --unlearn.lr=1e-5 --forgetting_set_size=1 --forgetting_set=[3] --unlearning_method=scrub
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.max_epochs=9 --unlearn.lr=1e-4 --forgetting_set_size=1 --forgetting_set=[3] --unlearning_method=badT
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.SSDdampening=2.5 --unlearn.SSDselectwt=0.001 --forgetting_set_size=1 --forgetting_set=[3] --unlearning_method=ssd
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.max_epochs=1000 --unlearn.lr=1e-4 --forgetting_set_size=1 --forgetting_set=[3] --unlearning_method=icus
python train_without_forgetting.py --dataset.name=cifar10 --model.name=resnet18 --train.max_epochs=15 --train.lr=1e-4 --train.batch_size=32 --forgetting_set_size=1 --forgetting_set=[3] --unlearning_method=finetuning
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.max_epochs=1000 --unlearn.lr=1e-4 --forgetting_set_size=2 --forgetting_set=[7] --unlearning_method=icus
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.scrub_steps=7 --unlearn.lr=1e-5 --forgetting_set_size=2 --forgetting_set=[7] --unlearning_method=scrub
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.max_epochs=9 --unlearn.lr=1e-4 --forgetting_set_size=2 --forgetting_set=[7] --unlearning_method=badT
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.SSDdampening=2.5 --unlearn.SSDselectwt=0.001 --forgetting_set_size=1 --forgetting_set=[7] --unlearning_method=ssd
python train_without_forgetting.py --dataset.name=cifar10 --model.name=resnet18 --train.max_epochs=15 --train.lr=1e-4 --train.batch_size=32 --forgetting_set_size=1 --forgetting_set=[7] --unlearning_method=finetuning
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.max_epochs=1000 --unlearn.lr=1e-4 --forgetting_set_size=2 --forgetting_set=[9] --unlearning_method=icus
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.scrub_steps=7 --unlearn.lr=1e-5 --forgetting_set_size=2 --forgetting_set=[9] --unlearning_method=scrub
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.max_epochs=9 --unlearn.lr=1e-4 --forgetting_set_size=2 --forgetting_set=[9] --unlearning_method=badT
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.SSDdampening=2.5 --unlearn.SSDselectwt=0.001 --forgetting_set_size=1 --forgetting_set=[9] --unlearning_method=ssd
python train_without_forgetting.py --dataset.name=cifar10 --model.name=resnet18 --train.max_epochs=15 --train.lr=1e-4 --train.batch_size=32 --forgetting_set_size=1 --forgetting_set=[9] --unlearning_method=finetuning
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.max_epochs=1000 --unlearn.lr=1e-4 --forgetting_set_size=2 --forgetting_set=[1,8] --unlearning_method=icus
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.scrub_steps=7 --unlearn.lr=1e-5 --forgetting_set_size=2 --forgetting_set=[1,8] --unlearning_method=scrub
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.max_epochs=9 --unlearn.lr=1e-4 --forgetting_set_size=2 --forgetting_set=[1,8] --unlearning_method=badT
python train_without_forgetting.py --dataset.name=cifar10 --model.name=resnet18 --train.max_epochs=15 --train.lr=1e-4 --train.batch_size=32 --forgetting_set_size=2 --forgetting_set=[1,8] --unlearning_method=finetuning
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.max_epochs=1000 --unlearn.lr=1e-4 --forgetting_set_size=2 --forgetting_set=[2,4] --unlearning_method=icus
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.scrub_steps=7 --unlearn.lr=1e-5 --forgetting_set_size=2 --forgetting_set=[2,4] --unlearning_method=scrub
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.max_epochs=9 --unlearn.lr=1e-4 --forgetting_set_size=2 --forgetting_set=[2,4] --unlearning_method=badT
python train_without_forgetting.py --dataset.name=cifar10 --model.name=resnet18 --train.max_epochs=15 --train.lr=1e-4 --train.batch_size=32 --forgetting_set_size=2 --forgetting_set=[2,4] --unlearning_method=finetuning
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.max_epochs=1000 --unlearn.lr=1e-4 --forgetting_set_size=2 --forgetting_set=[6,9] --unlearning_method=icus
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.scrub_steps=7 --unlearn.lr=1e-5 --forgetting_set_size=2 --forgetting_set=[6,9] --unlearning_method=scrub
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.max_epochs=9 --unlearn.lr=1e-4 --forgetting_set_size=2 --forgetting_set=[6,9] --unlearning_method=badT
python train_without_forgetting.py --dataset.name=cifar10 --model.name=resnet18 --train.max_epochs=15 --train.lr=1e-4 --train.batch_size=32 --forgetting_set_size=2 --forgetting_set=[6,9] --unlearning_method=finetuning
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.max_epochs=1000 --unlearn.lr=1e-4 --forgetting_set_size=3 --forgetting_set=[1,4,5] --unlearning_method=icus
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.scrub_steps=7 --unlearn.lr=1e-5 --forgetting_set_size=3 --forgetting_set=[1,4,5] --unlearning_method=scrub
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.max_epochs=6 --unlearn.lr=1e-4 --forgetting_set_size=3 --forgetting_set=[1,4,5] --unlearning_method=badT
python train_without_forgetting.py --dataset.name=cifar10 --model.name=resnet18 --train.max_epochs=15 --train.lr=1e-4 --train.batch_size=32 --forgetting_set_size=3 --forgetting_set=[1,4,5] --unlearning_method=finetuning
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.max_epochs=1000 --unlearn.lr=1e-4 --forgetting_set_size=3 --forgetting_set=[3,4,5] --unlearning_method=icus
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.scrub_steps=7 --unlearn.lr=1e-5 --forgetting_set_size=3 --forgetting_set=[3,4,5] --unlearning_method=scrub
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.max_epochs=6 --unlearn.lr=1e-4 --forgetting_set_size=3 --forgetting_set=[3,4,5] --unlearning_method=badT
python train_without_forgetting.py --dataset.name=cifar10 --model.name=resnet18 --train.max_epochs=15 --train.lr=1e-4 --train.batch_size=32 --forgetting_set_size=3 --forgetting_set=[3,4,5] --unlearning_method=finetuning
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.max_epochs=1000 --unlearn.lr=1e-4 --forgetting_set_size=3 --forgetting_set=[0,6,8] --unlearning_method=icus
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.scrub_steps=9 --unlearn.lr=1e-5 --forgetting_set_size=3 --forgetting_set=[0,6,8] --unlearning_method=scrub
python unlearn.py --dataset.name=cifar10 --model=resnet18 --unlearn.max_epochs=9 --unlearn.lr=1e-4 --forgetting_set_size=3 --forgetting_set=[0,6,8] --unlearning_method=badT
python train_without_forgetting.py --dataset.name=cifar10 --model.name=resnet18 --train.max_epochs=15 --train.lr=1e-4 --train.batch_size=32 --forgetting_set_size=3 --forgetting_set=[0,6,8] --unlearning_method=finetuning


