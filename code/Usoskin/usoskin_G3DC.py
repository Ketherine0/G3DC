import os
import torch
from sklearn import manifold, metrics
from visdom import Visdom

#tf.disable_v2_behavior()
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Parameter
from sklearn.cluster import KMeans
from metrics import *
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import pandas as pd
import sys
sys.path.append("..")
from  preprocess import load_data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(25334, 2048),
            # nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            # nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, 64),
            # nn.BatchNorm1d(512),
        )
        #   embedding: 100, 200
        self.decoder = nn.Sequential(
            # nn.BatchNorm1d(512),
            nn.Linear(64, 2048),
            # nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            # nn.BatchNorm1d(2048),
            nn.ReLU(True),
            nn.Linear(2048, 25334))
        self.model = nn.Sequential(self.encoder, self.decoder)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.model(x)
        return x


class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters=4, hidden=64, cluster_centers=None, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.n_clusters,
                self.hidden,
                dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers.to(x.device)) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        t_dist = (numerator.t() / torch.sum(numerator, 1)).t()  # soft assignment using t-distribution
        return t_dist


class DEC(nn.Module):
    def __init__(self, n_clusters=4, autoencoder=None, hidden=64, cluster_centers=None, alpha=1.0):
        super(DEC, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        self.cluster_centers = cluster_centers
        self.autoencoder = autoencoder
        self.clusteringlayer = ClusteringLayer(self.n_clusters, self.hidden, self.cluster_centers, self.alpha)

    def target_distribution(self, q_):
        weight = (q_ ** 2) / torch.sum(q_, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def forward(self, x):
        x = self.autoencoder.encode(x)
        return self.clusteringlayer(x)

    def visualize(self, epoch, x):
        fig = plt.figure()
        ax = plt.subplot(111)
        x = self.autoencoder.encode(x).detach()
        x = x.numpy()
        x_embedded = TSNE(n_components=2).fit_transform(x)
        plt.scatter(x_embedded[:, 0], x_embedded[:, 1])
        fig.savefig('plots_test/mnist_{}.png'.format(epoch))
        plt.close(fig)


class MyLoss(nn.Module):
    def __init__(self, l1, l2, L):
        super(MyLoss, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.L = torch.tensor(L)
        return

    def forward(self, W):

        L_norm = self.L.float()
        parameters = list(autoencoder.parameters())
        W = torch.tensor(parameters[0].detach().numpy(), requires_grad=True)
        tr = torch.trace(torch.mm(torch.mm(W, L_norm), torch.transpose(W, 0, 1)))
        print('tr loss',tr)

        W1 = torch.transpose(W, 0, 1)
        L21_norm = (torch.sqrt(torch.mul(W1, W1).sum(1))).sum()
        print('L21 loss',L21_norm)

        loss_sum = self.l1*tr + self.l2*L21_norm
        return loss_sum


def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving new checkpoint")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")


def pretrain(**kwargs):
    data = kwargs['data']
    model = kwargs['model']
    num_epochs = kwargs['num_epochs']
    savepath = kwargs['savepath']
    checkpoint = kwargs['checkpoint']
    start_epoch = checkpoint['epoch']
    parameters = list(autoencoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3,weight_decay=1e-5)
    train_loader = DataLoader(dataset=data,
                              batch_size=610,
                              shuffle=True)
    row=[]
    for epoch in range(start_epoch, num_epochs):
        for data in train_loader:
            img = data.float()
            img = img.to(device)
            # ===================forward=====================
            output = model(img)
            out = output.argmax(1)
            output = output.view(output.size(0), 25334)
            loss = nn.MSELoss()(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        #     accuracy = acc(y.cpu().numpy(), out.cpu().numpy())
            accuracy = metrics.adjusted_rand_score(y.cpu().numpy(), out.cpu().numpy())
            row.append([epoch, accuracy])
            print('Epochs: [{}/{}], Loss:{}'.format(epoch, num_epochs, loss))
        state = loss.item()
        is_best = False
        if state < checkpoint['best']:
            checkpoint['best'] = state
            is_best = True

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best': state,
            'epoch': epoch
        }, savepath,
            is_best)


def train(**kwargs):
    data = kwargs['data']
    labels = kwargs['labels']
    model = kwargs['model']
    num_epochs = kwargs['num_epochs']
    savepath = kwargs['savepath']
    checkpoint = kwargs['checkpoint']
    start_epoch = checkpoint['epoch']
    parameters = list(autoencoder.parameters())
    features = []
    train_loader = DataLoader(dataset=data,
                              batch_size=610,
                              shuffle=False)

    for i, batch in enumerate(train_loader):
        img = batch.float()
        img = img.to(device)
        features.append(model.autoencoder.encode(img).detach())
    features = torch.cat(features)
    # ============K-means=======================================
    kmeans = KMeans(n_clusters=4, random_state=0).fit(features.cpu())
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float)
    model.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)
    # =========================================================
    y_pred = kmeans.predict(features.cpu())
    accuracy = normalized_mutual_info_score(y.numpy(), y_pred)
    print('Initial Accuracy: {}'.format(accuracy))

    lambda1 = torch.tensor(1e-2)
    lambda2 = torch.tensor(1e-4)

    W = torch.tensor(parameters[0].detach().numpy(), requires_grad=True)

    a = MyLoss(l1=lambda1, l2=lambda2, L=adj)
    loss_function = nn.KLDivLoss(size_average=False)

    optimizer2 = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.99, weight_decay=1e-7)

    print('Training')
    row = []
    for epoch in range(start_epoch, num_epochs):
        batch = data
        img = batch.float()
        img = img.to(device)
        # output of t-dist
        # ===================forward=====================
        output = model(img)
        reconst_output = autoencoder(img)

        reconst_output = reconst_output.view(reconst_output.size(0), 25334)
        loss_mse = nn.MSELoss()(reconst_output, img)

        if epoch % 10 == 0:
            W1 = torch.tensor(parameters[0].detach().numpy(), requires_grad=True)
            np.savetxt('%s %s %s' % ('model_weight/usoskin_weight/layer_weight/W2', epoch, '.txt'), W1.cpu().detach().numpy())
        # output of target distribution
        target = model.target_distribution(output).detach()
        out = output.argmax(1)

        ''' put weight '''

        z = []
        for i, batch in enumerate(train_loader):
            img = batch.float()
            img = img.to(device)
            z.append(model.autoencoder.encode(img).detach())
        W1 = parameters[0]
        loss1 = a(W1)
        loss2 = loss_function(torch.log(output), target)
        gamma = 5

        loss = gamma*loss_mse + loss2+loss1
        print('mse_loss',loss_mse)
        print('div_origin loss',loss2)

        # loss = loss_function(output.log(), target) / output.shape[0]
        ''' revise loss '''

        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()

        accuracy = acc(y.cpu().numpy(), out.cpu().numpy())
        NMI = metrics.normalized_mutual_info_score(y.cpu().numpy(), out.cpu().numpy())
        ARI = adjusted_rand_score(y.cpu().numpy(), out.cpu().numpy())
        row.append([epoch, accuracy,NMI,ARI])
        print('Epochs: [{}/{}] Accuracy:{}, Loss:{}'.format(epoch, num_epochs, accuracy, loss))
        print('NMI:{}, Loss:{}'.format(NMI, loss))
        print('ARI:{}, Loss:{}'.format(ARI, loss))

        state = loss.item()
        is_best = False
        if state < checkpoint['best']:
            checkpoint['best'] = state
            is_best = True

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best': state,
            'epoch': epoch
        }, savepath,
            is_best)

    df = pd.DataFrame(row, columns=['epochs', 'accuracy','NMI','ARI'])
    df.to_csv('usoskin_W/log_acc.csv')




if __name__ == '__main__':
    x, y, adj, gene_name_li, edge_li, a = load_data()
    x = torch.tensor(x)
    y = torch.tensor(y)

    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', default=610, type=int)
    parser.add_argument('--pretrain_epochs', default=300, type=int)
    parser.add_argument('--train_epochs', default=40, type=int)
    parser.add_argument('--save_dir', default='model_weight/usoskin_weight/path')
    args = parser.parse_args()

    print(args)
    epochs_pre = args.pretrain_epochs
    batch_size = args.batch_size

    autoencoder = AutoEncoder().to(device)
    ae_save_path = 'model_weight/usoskin_weight/path/sim_autoencoder3.pth'

    if os.path.isfile(ae_save_path):
        print('Loading {}'.format(ae_save_path))
        checkpoint = torch.load(ae_save_path)
        autoencoder.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(ae_save_path))
        checkpoint = {
            "epoch": 0,
            "best": float("inf")
        }
    pretrain(data=x, model=autoencoder, num_epochs=epochs_pre, savepath=ae_save_path, checkpoint=checkpoint)

    dec_save_path = 'model_weight/usoskin_weight/path/dec6.pth'
    dec = DEC(n_clusters=4, autoencoder=autoencoder, hidden=64, cluster_centers=None, alpha=1.0).to(device)
    if os.path.isfile(dec_save_path):
        print('Loading {}'.format(dec_save_path))
        checkpoint = torch.load(dec_save_path)
        dec.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(dec_save_path))
        checkpoint = {
            "epoch": 0,
            "best": float("inf")
        }

    train(data=x, labels=y, model=dec, num_epochs=args.train_epochs, savepath=dec_save_path, checkpoint=checkpoint)



