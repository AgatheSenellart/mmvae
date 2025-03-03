
import numpy as np
from sklearn import svm
from tqdm import tqdm
import time
from pathlib import Path
import matplotlib.pyplot as plt


import torch
import wandb
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau


from bivae.dcca.linear_cca import linear_cca
from bivae.dataloaders import MEDMNIST_DL
from bivae.dcca.models import DeepCCA_MedMNIST
from bivae.dcca.utils import  svm_classify_view, unpack_data, visualize_umap, save_encoders
from torchvision import transforms
from bivae.utils import add_channels

torch.set_default_tensor_type(torch.DoubleTensor)



class Solver():
    def __init__(self, model, linear_cca, outdim_size, epoch_num, batch_size, learning_rate, reg_par, device=torch.device('cpu')):
        self.model = model
        self.model.to(device)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.loss = model.loss
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=learning_rate, weight_decay=reg_par)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        self.device = device

        self.linear_cca = linear_cca

        self.outdim_size = outdim_size

        
        print('Solver initialized')

    def fit(self, train_loader, val_loader=None, tx1=None, tx2=None, checkpoint=None):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        print('Starting the optimization')


        if val_loader is not None:
            best_val_loss = 0
            
        num_epochs_without_improvement =0
        train_losses = []
        for epoch in range(self.epoch_num):
            epoch_start_time = time.time()
            self.model.train()

            for dataT in tqdm(train_loader):
                data = unpack_data(dataT)
                self.optimizer.zero_grad()
                batch_x1 = data[0]
                batch_x2 = data[1]
                o1, o2 = self.model(batch_x1, batch_x2)
                loss = self.loss(o1, o2)
                train_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            train_loss = np.mean(train_losses)
            wandb.log({'train_loss' : train_loss})
            info_string = "Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f}"
            if val_loader is not None:
                with torch.no_grad():
                    self.model.eval()
                    val_loss = self.test(val_loader)[0]
                    wandb.log({'val_loss' : val_loss})
                    info_string += " - val_loss: {:.4f}".format(val_loss)
                    self.scheduler.step(val_loss)
                    if val_loss < best_val_loss:
                        print(
                            "Epoch {:d}: val_loss improved from {:.4f} to {:.4f}, saving model to {}".format(epoch + 1, best_val_loss, val_loss, checkpoint))
                        best_val_loss = val_loss
                        save_encoders(model,checkpoint)
                        num_epochs_without_improvement = 0 # reset early stop
                    else:
                        print("Epoch {:d}: val_loss did not improve from {:.4f}".format(
                            epoch + 1, best_val_loss))
                        num_epochs_without_improvement+=1
            else:
                save_encoders(model, checkpoint)
            epoch_time = time.time() - epoch_start_time
            print((info_string.format(
                epoch + 1, self.epoch_num, epoch_time, train_loss)))

            if num_epochs_without_improvement == 20:
                break
        # train_linear_cca
        if self.linear_cca is not None:
            _, outputs = self._get_outputs(train_loader)
            self.train_linear_cca(outputs[0], outputs[1])


        if val_loader is not None:
            loss = self.test(val_loader)[0]
            print("loss on validation data: {:.4f}".format(loss))

        

    def test(self, test_loader, use_linear_cca=False):
        with torch.no_grad():
            losses, outputs = self._get_outputs(test_loader)

            if use_linear_cca:
                print("Linear CCA started!")
                outputs_cca = self.linear_cca.test(outputs[0], outputs[1])
                return np.mean(losses), [*outputs_cca,outputs[-1]]
            else:
                return np.mean(losses), outputs

    def train_linear_cca(self, x1, x2):
        self.linear_cca.fit(x1, x2, self.outdim_size)

    def _get_outputs(self, test_loader):
        with torch.no_grad():
            self.model.eval()
            data_size = len(train_loader.dataset)
            batch_idxs = list(BatchSampler(SequentialSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            losses = []
            outputs1 = []
            outputs2 = []
            labels = []
            for dataT in test_loader:
                labels.append(dataT[0][1])
                data = unpack_data(dataT)
                batch_x1 = data[0]
                batch_x2 = data[1]
                o1, o2 = self.model(batch_x1, batch_x2)
                outputs1.append(o1)
                outputs2.append(o2)
                loss = self.loss(o1, o2)
                losses.append(loss.item())
        outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                   torch.cat(outputs2, dim=0).cpu().numpy(),
                   torch.cat(labels, dim=0).cpu().numpy()]
        return losses, outputs


if __name__ == '__main__':

    print(torch.cuda.is_available())


    ############
    # Parameters Section

    device = torch.device('cuda')
    print("Using", torch.cuda.device_count(), "GPUs")

    # the path to save the models
    save_to = Path('../experiments/dcca/medmnist/')
    save_to.mkdir(parents=True, exist_ok=True)

    # the size of the new space learned by the model (number of the new features)
    outdim_size = 16


    # the parameters for training the network
    learning_rate = 1e-4
    epoch_num = 100
    batch_size = 500
    dlargs = {'drop_last':True}
    train_loader,test_loader, val_loader = MEDMNIST_DL().getDataLoaders(batch_size=batch_size, shuffle=True,dlargs=dlargs)


    # the regularization parameter of the network
    # seems necessary to avoid the gradient exploding especially when non-saturating activations are used
    reg_par = 1e-5

    # specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
    # if one option does not work for a network or dataset, try the other one
    use_all_singular_values = False

    # if a linear CCA should get applied on the learned features extracted from the networks
    # it does not affect the performance on noisy MNIST significantly
    apply_linear_cca = True
    # end of parameters section
    ############
    wandb.init(project = 'dcca_medmnist', entity = 'asenellart', config = {'batch_size' : batch_size,
                                                                            'learning_rate': learning_rate,
                                                                            'reg_par' : reg_par,
                                                                            'linear_cca' : linear_cca is not None, 
                                                                            'outdim_size':outdim_size,
                                                                            'use_all_singular_values': use_all_singular_values},
                )

    # Building, training, and producing the new features by DCCA
    model = DeepCCA_MedMNIST(outdim_size, use_all_singular_values, device=device).double()
    l_cca = None
    if apply_linear_cca:
        l_cca = linear_cca()
    solver = Solver(model, l_cca, outdim_size, epoch_num, batch_size,
                    learning_rate, reg_par, device=device)

    solver.fit(train_loader,val_loader, checkpoint=save_to)
    # TODO: Save l_cca model if needed
    if apply_linear_cca :
        np.save(str(save_to) + '/l_cca_w.npy', l_cca.w)
        np.save(str(save_to) + '/l_cca_m.npy', l_cca.m)
        np.save(str(save_to) + '/l_cca_D.npy', l_cca.D)
        
        # Visualize the singular values to gauge if the outdimsize is well chosen 
        plt.plot(l_cca.D)
        plt.savefig(str(save_to) + '/singular_values.png')
        plt.close()
        wandb.log({'Singular Values of DCCA' : wandb.Image(str(save_to) + '/singular_values.png')})
        wandb.log({'total_dcca_value' : np.mean(l_cca.D[:outdim_size])})

    # Training and testing of SVM with linear kernel on the view 1 with new features
    losses, outputs_t =  solver.test(train_loader, use_linear_cca=apply_linear_cca)
    losses, outputs_s = solver.test(test_loader, use_linear_cca=apply_linear_cca)
    
    visualize_umap(outputs_s[1], outputs_s[2], save_file=str(save_to) + '/embedding_1.png')
    visualize_umap(outputs_s[0], outputs_s[2], save_file=str(save_to) + '/embedding_2.png')

    wandb.log({'embedding_svhn' : wandb.Image(str(save_to) + '/embedding_1.png')})
    wandb.log({'embedding_mnist' : wandb.Image(str(save_to) + '/embedding_2.png')})
    for i in range(2):
        test_acc = svm_classify_view(outputs_t, outputs_s, C=0.01, view=i)
        print(f"Accuracy on view {i} (test data) is:", test_acc*100.0)

    