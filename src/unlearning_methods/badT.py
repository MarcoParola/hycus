import torch, copy
import torch.nn.functional as F
import torch.nn as nn
import tqdm
from torch.cuda.amp import autocast, GradScaler
from src.unlearning_methods.base import BaseUnlearningMethod
#STUFF TO BE TESTED
from torch.optim.lr_scheduler import ReduceLROnPlateau


class BadT(BaseUnlearningMethod):
    def __init__(self, opt, model, forgetting_subset, logger=None):
        super().__init__(opt, model)
        print("BadT initialization")
        self.og_model = copy.deepcopy(model)  # copy the model
        self.og_model.to(self.opt.device)
        self.og_model.eval()  # put the model in evaluation mode
        self.forgetting_subset = forgetting_subset
        self.logger=logger
        
        # create a random model
        self.random_model = copy.deepcopy(model)
        self.random_model.apply(self._random_weights_init)
        self.random_model.to(self.opt.device)
        self.random_model.eval() 
        
        # Initialize the optimizer, scheduler and scaler
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt.unlearn.lr, momentum=0.9, weight_decay=0.001)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=False)
        self.scaler = GradScaler()
        self.kltemp = opt.unlearn.temp  # Temperature for KL-divergenza (knowledge distillation)

    def _random_weights_init(self, model):
        if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear):
            torch.nn.init.xavier_uniform_(model.weight)
            if model.bias is not None:
                torch.nn.init.zeros_(model.bias)

    def forward_pass(self, sample, target, infgt):
        output = self.model(sample)
        
        # Calculate logits (original e random)
        full_teacher_logits = self.og_model(sample)
        unlearn_teacher_logits = self.random_model(sample)
        
        # Apply softmax 
        f_teacher_out = torch.nn.functional.softmax(full_teacher_logits / self.kltemp, dim=1)
        u_teacher_out = torch.nn.functional.softmax(unlearn_teacher_logits / self.kltemp, dim=1)
        
        # Select which output to use depending from infgt
        labels = torch.unsqueeze(infgt, 1)
        overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
        
        # Calculate loss with KL-divergence
        student_out = F.log_softmax(output / self.kltemp, dim=1)
        loss = F.kl_div(student_out, overall_teacher_out)
        return output,loss



    def train_one_epoch(self, loader):
            self.model.train()  # Set the model in training mode

            for inputs, labels, infgt in tqdm.tqdm(loader):
                inputs, labels, infgt = inputs.to(self.opt.device), labels.to(self.opt.device), infgt.to(self.opt.device)
                with autocast(): 
                    # reset gradients
                    self.optimizer.zero_grad()
                    # Execute the forward pass
                    preds, loss = self.forward_pass(inputs, labels, infgt)
                    self.logger.log_metrics({"method":"BadT", "loss": loss.item()}, step=self.curr_step)
                    self.scaler.scale(loss).backward()  # forward pass and backpropagation
                    self.scaler.step(self.optimizer) #update weights
                    self.scaler.update() # update scaler
            self.scheduler.step(loss) # update learning rate
            print(f'Epoch: {self.epoch}')
            return
