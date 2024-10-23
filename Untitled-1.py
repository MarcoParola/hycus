

classifier_model = resnet9(..)
unlearning_model = icus(..)
unlearned_classifier_model = resnet9(..)

train, val, test = load_dataset()
w_train, w_val, w_test = load_wrapped_dataset()

for i in range(n_classes):
    w, descr = w_val[i]

    latent = unlearning_model.ae1(w)
    w_new, bias_new = unlearning_model.ae2(latent)

    latent = unlearning_model.ae3(descr)
    w_new, bias_new = unlearning_model.ae2(latent)

    unlearned_classifier_model.fc[i].weight = w_new
    unlearned_classifier_model.fc[i].bias = bias_new


#Step 1: 
