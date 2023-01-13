import os
import csv
import numpy as np
import scipy.sparse as sp
#import tensorflow.compat.v1 as tf
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
from singleCell3 import load_data
from sklearn import metrics
#from sgd import SGD

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# python test.py "example_expression.csv" "example_adjacency.txt" "var_impo.csv"

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(14141, 2048),
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
            nn.Linear(2048, 14141))
        self.model = nn.Sequential(self.encoder, self.decoder)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.model(x)
        return x


class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters=8, hidden=64, cluster_centers=None, alpha=1.0):
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
    def __init__(self, n_clusters=8, autoencoder=None, hidden=64, cluster_centers=None, alpha=1.0):
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
        L = np.array(L)
        self.L = torch.tensor(L)
        return

    def forward(self, W):

        L_norm = self.L.float()
        # tr = trace(W, L)
        # nrow = W.shape[0]
        # L21_nrom = 0
        # for i in range(nrow):
        #     W2 = np.linalg.norm(W[i], ord=2, axis=None, keepdims=False)
        #     L21_nrom += W2
        # W: 800x1000

        tr = torch.trace(torch.mm(torch.mm(W, L_norm), torch.transpose(W, 0, 1)))
        print('tr loss',tr)

        W1 = torch.transpose(W, 0, 1)
        L21_norm = (torch.sqrt(torch.mul(W1, W1).sum(1))).sum()

        # print(torch.sqrt(torch.mm(W, torch.transpose(W,0,1)).sum(1)))
        # L21_norm = L21(W.data)
        print('L21 loss',L21_norm)

        loss_sum = self.l1*tr + self.l2*L21_norm
        # loss_sum = self.l2*L21_norm
        # np.savetxt("W_usoskin.txt", W.cpu().detach().numpy())
        # loss_sum.backward()
        # print('W grad',W.grad,W.grad_fn)
        return loss_sum


def add_noise(img):
    noise = torch.randn(img.size()) * 0.2
    noisy_img = img + noise
    return noisy_img


def save_checkpoint(state, filename, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving new checkpoint")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")


def t_SNE(output, dimention):
    # output:待降维的数据
    # dimention：降低到的维度
    tsne = manifold.TSNE(n_components=dimention, init='pca', random_state=0)
    result = tsne.fit_transform(output)
    return result

# Visualization with visdom
def Visualization(vis, result, labels,title):
    # vis: Visdom对象
    # result: 待显示的数据，这里为t_SNE()函数的输出
    # label: 待显示数据的标签
    # title: 标题
    vis.scatter(
        X = result,
        Y = labels+1,           # 将label的最小值从0变为1，显示时label不可为0
       opts=dict(markersize=4,title=title),
    )


def pretrain(**kwargs):
    data = kwargs['data']
    model = kwargs['model']
    num_epochs = kwargs['num_epochs']
    savepath = kwargs['savepath']
    checkpoint = kwargs['checkpoint']
    start_epoch = checkpoint['epoch']
    parameters = list(autoencoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3,weight_decay=1e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_loader = DataLoader(dataset=data,
                              batch_size=2180,
                              shuffle=True)
    row=[]
    for epoch in range(start_epoch, num_epochs):
        for data in train_loader:
            img = data.float()
            noisy_img = add_noise(img)
            noisy_img = noisy_img.to(device)
            img = img.to(device)
            # ===================forward=====================
            output = model(img)
            # output of target distribution
            # print('tar',target)
            out = output.argmax(1)

            output = output.view(output.size(0), 14141)
            loss = nn.MSELoss()(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        # accuracy = metrics.adjusted_rand_score(y.cpu().numpy(), out.cpu().numpy())
        #     accuracy = acc(y.cpu().numpy(), out.cpu().numpy())
            accuracy = normalized_mutual_info_score(y.cpu().numpy(), out.cpu().numpy())
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
                              batch_size=2180,
                              shuffle=True)

    for i, batch in enumerate(train_loader):
        img = batch.float()
        img = img.to(device)
        features.append(model.autoencoder.encode(img).detach())
    features = torch.cat(features)
    # ============K-means=======================================
    kmeans = KMeans(n_clusters=8, random_state=0).fit(features.cpu())
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float)
    model.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)
    # =========================================================
    y_pred = kmeans.predict(features.cpu())
    # print(y)
    # print(y_pred)
    accuracy = acc(y.numpy(), y_pred)
    # accuracy = acc(y.numpy(), y_pred)
    # accuracy = metrics.normalized_mutual_info_score(y.numpy(), y_pred)
    print('Initial Accuracy: {}'.format(accuracy))

    lambda1 = torch.tensor(1e-5)
    lambda2 = torch.tensor(1e-3)
    # loss_function = Reg_loss(size_average=False, l1=lambda1, l2=lambda2, l3=lambda3)
    # loss_function = nn.KLDivLoss(size_average=False)

    W = torch.tensor(parameters[0].detach().numpy(), requires_grad=True)

    a = MyLoss(l1=lambda1, l2=lambda2, L=adj)
    loss_function = nn.KLDivLoss(size_average=False)

    # model_param = model.parameters
    # print('model para', model_param)

    # para_dict = {{lambda1:lambda1}, {lambda2:lambda2}, model_param}
    # print('para_dict:', para_dict)

    # optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.99)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-7)
    # First stage:
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.99, weight_decay=1e-5)
    # Second:
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-5)
    print('Training')
    row = []
    for epoch in range(start_epoch, num_epochs):
        batch = data
        img = batch.float()
        img = img.to(device)
        # output of t-dist

        output = model(img)
        reconst_output = autoencoder(img)

        reconst_output = reconst_output.view(reconst_output.size(0), 14141)
        loss_mse = nn.MSELoss()(reconst_output, img)

        if epoch % 10 == 0:
            W1 = torch.tensor(parameters[0].detach().numpy(), requires_grad=True)
            # np.savetxt('%s %s %s' % ('pbmcStim_test_5_tsne3_W/W3', epoch, '.txt'), W1.cpu().detach().numpy())
        # if epoch % 5 == 0:
        #     np.savetxt('%s %s %s' % ('pbmcStim_test_5_tsne3_W/feature3', epoch, '.txt'), features)
        # output of target distribution
        target = model.target_distribution(output).detach()
        # print('tar',target)
        out = output.argmax(1)
        # if epoch % 20 == 0:
        #     print('plotting')
        #     dec.visualize(epoch, img)

        ''' put weight '''

        # W1 = torch.tensor(parameters[0].detach().numpy(), requires_grad=True)
        # img: 100x500 (input)
        # cluster_center =
        # z: 100x200
        # z.unsqueeze(1): 100x1x200
        z = []
        for i, batch in enumerate(train_loader):
            img = batch.float()
            img = img.to(device)
            z.append(model.autoencoder.encode(img).detach())
        # z = torch.cat(z)
        # mu = cluster_centers
        # z = torch.tensor(z,requires_grad=True)
        # mu = torch.tensor(mu,requires_grad=True)
        W1 = parameters[0]
        loss1 = a(W1)
        # loss2 = loss_function(output.log(), target) / output.shape[0]
        loss2 = loss_function(torch.log(output), target)

        gamma1 = 0.01 * loss2 / loss_mse

        loss = gamma1*loss_mse + loss2+loss1
        print('mse_loss',gamma1*loss_mse)
        print('div_origin loss',loss2)

        # loss = loss_function(output.log(), target) / output.shape[0]
        ''' revise loss '''

        optimizer.zero_grad()
        loss.backward()
        # print('loss grad',loss.grad)
        optimizer.step()

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
    df.to_csv('pbmcStim_test_5_tsne3_W/log_acc3.csv')
    df1 = pd.DataFrame(features)
    # df1.to_csv('pbmcStim_test_5_tsne3_W/feature3.csv')

#  Visualization
    # python -m visdom.server
    # http://localhost:8097/#
    # output = output.detach().numpy()
    # labels = labels.detach().numpy()
    # preds = out.detach().numpy()

    # Visualization with visdom
    # vis = Visdom(env='pyGCN Visualization')
    if False:
        vis = Visdom(env='iDEC SingleCell Visualization')

        result_all_2d = t_SNE(features, 2)
        Visualization(vis, result_all_2d, labels,
                      title='[Output of pbmcCtrl]\n Dimension reduction to %dD' % (
                          result_all_2d.shape[1]))
        result_all_3d = t_SNE(features, 3)
        Visualization(vis, result_all_3d, labels,
                      title='[Output of pbmcCtrl]\n Dimension reduction to %dD' % (
                          result_all_3d.shape[1]))

        print('Finished')




if __name__ == '__main__':
    # n = 100  p = 500
    x, y, adj, gene_name, edge_li, a = load_data()
    # x, y, adj = load_data()
    # print('adj',adj)
    x = torch.tensor(x)
    y = torch.tensor(y)

    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', default=2180, type=int)
    parser.add_argument('--pretrain_epochs', default=900, type=int)
    parser.add_argument('--train_epochs', default=100, type=int)
    parser.add_argument('--save_dir', default='pbmcStim_test_5_tsne3')
    args = parser.parse_args()
    print(args)
    epochs_pre = args.pretrain_epochs
    batch_size = args.batch_size

    autoencoder = AutoEncoder().to(device)
    ae_save_path = 'pbmcStim_test_5_tsne3/sim_autoencoder2.pth'

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

    dec_save_path = 'pbmcStim_test_5_tsne3/dec4.pth'
    dec = DEC(n_clusters=8, autoencoder=autoencoder, hidden=64, cluster_centers=None, alpha=1.0).to(device)
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



