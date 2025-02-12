python scripts/confusion_matrix.py --dataset.name=lfw --dataset.classes=62 --forgetting_set_size=3 --forgetting_set=[12,34,48]
python test.py --dataset.name=cifar100 --dataset.classes=100 --model.name=resnet18 --forgetting_set_size=3 --forgetting_set=[12,34,48] --golden_model=True
python test.py --dataset.name=cifar100 --dataset.classes=100 --model.name=resnet18 --forgetting_set_size=3 --forgetting_set=[12,34,48] --unlearning_method=icus
python test.py --dataset.name=cifar100 --dataset.classes=100 --model.name=resnet18 --forgetting_set_size=3 --forgetting_set=[12,34,48] --unlearning_method=badT
python test.py --dataset.name=cifar100 --dataset.classes=100 --model.name=resnet18 --forgetting_set_size=3 --forgetting_set=[12,34,48] --unlearning_method=scrub
python test.py --dataset.name=cifar100 --dataset.classes=100 --model.name=resnet18 --forgetting_set_size=3 --forgetting_set=[12,34,48] --unlearning_method=finetuning