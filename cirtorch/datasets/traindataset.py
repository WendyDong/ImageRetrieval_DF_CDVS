import os
import pickle
import pdb
import torch
import torch.utils.data as data
import random
import numpy as np
from cirtorch.datasets.datahelpers import default_loader, imresize, cid2filename
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.utils.general import get_data_root
from PIL import Image

class TuplesDataset(data.Dataset):
    """Data loader that loads training and validation tuples

    Args:
        name (string): dataset name: 'retrieval-sfm-120k'
        mode (string): 'train' or 'val' for training and validation parts of dataset
        imsize (int, Default: None): Defines the maximum size of longer image side
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        loader (callable, optional): A function to load an image given its path.
        nnum (int, Default:5): Number of negatives for a query image in a training tuple
        qsize (int, Default:2000): Number of query images, ie number of (q,p,n1,...nN) tuples, to be processed in one epoch
        poolsize (int, Default:2000): Pool size for negative images re-mining
    """

    def __init__(self, name, mode, imsize=None, nnum=5, qsize=2000, poolsize=2000, transform=None, sample_diy=False, using_cdvs=0,
                 loader=default_loader):

        if not (mode == 'train' or mode == 'val'):
            raise (RuntimeError("MODE should be either train or val, passed as string"))
        self.using_cdvs=using_cdvs
        if name.startswith('retrieval-SfM'):
            # setting up paths
            data_root = get_data_root()
            db_root = os.path.join(data_root, 'train', name)
            ims_root = os.path.join(db_root, 'ims')

            # loading db
            db_fn = os.path.join(db_root, '{}.pkl'.format(name+'-my'))
            with open(db_fn, 'rb') as f:
                db = pickle.load(f)[mode]

            # setting fullpath for images
            self.images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]
            if self.using_cdvs!=0:
                self.global_cdvs=db['cids_cdvs']
        elif name.startswith('cdvs_train_retrieval'):
            data_root = get_data_root()
            # loading db
            db_fn = os.path.join(data_root, 'train', '{}.pkl'.format(name))
            with open(db_fn, 'rb') as f:
                db = pickle.load(f)[mode]
            # setting fullpath for images
            self.images = [os.path.join(data_root, db['cids'][i]) for i in range(len(db['cids']))]

        else:
            raise (RuntimeError("Unknown dataset name!"))
        if self.using_cdvs != 0:
            self.global_cdvs = db['cids_cdvs']
        # initializing tuples dataset
        self.sample_diy=sample_diy
        self.name = name
        self.mode = mode
        self.imsize = imsize
        self.clusters = db['cluster']
        self.qpool = db['qidxs']
        self.ppool = db['pidxs']


        # size of training subset for an epoch
        self.nnum = nnum
        self.qsize = min(qsize, len(self.qpool))
        self.poolsize = min(poolsize, len(self.images))
        self.qidxs = None
        self.pidxs = None
        self.nidxs = None

        self.transform = transform
        self.loader = loader

        self.print_freq = 10

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            images tuple (q,p,n1,...,nN): Loaded train/val tuple at index of self.qidxs
        """
        if self.__len__() == 0:
            raise (RuntimeError(
                "List qidxs is empty. Run ``dataset.create_epoch_tuples(net)`` method to create subset for train/val!"))

        output = []
        # query image
        output.append(self.loader(self.images[self.qidxs[index]]))
        output.append(self.loader(self.images[self.pidxs[index]]))
        # negative images
        for i in range(len(self.nidxs[index])):
            output.append(self.loader(self.images[self.nidxs[index][i]]))

        if self.imsize is not None:
            output = [imresize(img, self.imsize) for img in output]

        if self.transform is not None:
            output = [self.transform(output[i]).unsqueeze_(0) for i in range(len(output))]

        target = torch.Tensor([-1, 1] + [0] * len(self.nidxs[index]))

        return output, target

    def __len__(self):
        # if not self.qidxs:
        #     return 0
        # return len(self.qidxs)
        return self.qsize

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Name and mode: {} {}\n'.format(self.name, self.mode)
        fmt_str += '    Number of images: {}\n'.format(len(self.images))
        fmt_str += '    Number of training tuples: {}\n'.format(len(self.qpool))
        fmt_str += '    Number of negatives per tuple: {}\n'.format(self.nnum)
        fmt_str += '    Number of tuples processed in an epoch: {}\n'.format(self.qsize)
        fmt_str += '    Pool size for negative remining: {}\n'.format(self.poolsize)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def create_epoch_tuples(self, net, temp_loss):

        print('>> Creating tuples for an epoch of {}-{}...'.format(self.name, self.mode))

        ## ------------------------
        ## SELECTING POSITIVE PAIRS
        ## ------------------------

        # draw qsize random queries for tuples
        idxs2qpool = torch.randperm(len(self.qpool))[:self.qsize]
        self.qidxs = [self.qpool[i] for i in idxs2qpool]
        self.pidxs = [self.ppool[i] for i in idxs2qpool]

        ## ------------------------
        ## SELECTING NEGATIVE PAIRS
        ## ------------------------

        # if nnum = 0 create dummy nidxs
        # useful when only positives used for training
        if self.nnum == 0:
            self.nidxs = [[] for _ in range(len(self.qidxs))]
            return 0

        # draw poolsize random images for pool of negatives images
        ####if using less data
        newimages = []
        for i in range(len(self.images)):
            if self.clusters[i] != -1:
                newimages.append(i)
        random.shuffle(newimages)
        idxs2images = newimages[:self.poolsize]

        # prepare network
        net.cuda()
        net.eval()
        batchsize = min(100, self.qsize)
        # no gradients computed, to reduce memory and increase speed
        with torch.no_grad():

            print('>> Extracting descriptors for query images...')
            # prepare query loader
            loader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=[self.images[i] for i in self.qidxs], imsize=self.imsize, re_size=True,
                               transform=self.transform),
                batch_size=batchsize, shuffle=False, num_workers=8, pin_memory=True
            )
            # extract query vectors
            qvecs = torch.zeros(net.meta['outputdim'], len(self.qidxs)).cuda()
            for i, input in enumerate(loader):
                if batchsize != 1:
                    if (i + 1) != len(loader):
                        qvecs[:, i * batchsize:(i + 1) * batchsize] = net(input.cuda()).data.squeeze()
                    else:
                        qvecs[:, i * batchsize:len(self.qidxs)] = net(input.cuda()).data.squeeze()
                else:
                    qvecs[:, i] = net(input.cuda()).data.squeeze()
                if (i + 1) % self.print_freq == 0 or (i + 1) == len(loader):
                    print('\r>>>> {}/{} done...'.format((i + 1), len(loader)), end='')
            print('')

            print('>> Extracting descriptors for negative pool...')
            # prepare negative pool data loader

            loader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=[self.images[i] for i in idxs2images], imsize=self.imsize, re_size=True,
                               transform=self.transform),
                batch_size=batchsize, shuffle=False, num_workers=8, pin_memory=True
            )
            # extract negative pool vectors
            poolvecs = torch.zeros(net.meta['outputdim'], len(idxs2images)).cuda()
            for i, input in enumerate(loader):
                if batchsize != 1:
                    if (i + 1) != len(loader):
                        poolvecs[:, i * batchsize:(i + 1) * batchsize] = net(input.cuda()).data.squeeze()
                    else:
                        poolvecs[:, i * batchsize:len(idxs2images)] = net(input.cuda()).data.squeeze()
                else:
                    poolvecs[:, i] = net(input.cuda()).data.squeeze()
                if (i + 1) % self.print_freq == 0 or (i + 1) == len(loader):
                    print('\r>>>> {}/{} done...'.format((i + 1), len(loader)), end='')

            #                 poolvecs[:, i] = net(input.cuda()).data.squeeze()
            # if (i + 1) % self.print_freq == 0 or (i + 1) == len(idxs2images):
            #     print('\r>>>> {}/{} done...'.format(i + 1, len(idxs2images)), end='')
            print('')
            print('>> Searching for hard negatives...')
            torch.cuda.empty_cache()
            # compute dot product scores and ranks on GPU
            scores = torch.mm(poolvecs.t(), qvecs)
            if self.using_cdvs!=0:
                print('using global descriptor')
                qvecs_global = torch.from_numpy(np.vstack([self.global_cdvs[:, i] for i in self.qidxs])).float().cuda().transpose(1,0)
                # qvecs = torch.cat((qvecs_global, qvecs), 0)

                poolvecs_global = torch.from_numpy(np.vstack([self.global_cdvs[:, i] for i in idxs2images])).float().cuda().transpose(1,0)
                # poolvecs = torch.cat((poolvecs_global, poolvecs), 0)
                scores_global = torch.mm(poolvecs_global.t(), qvecs_global)
                if self.using_cdvs!=-1:
                    scores=scores+scores_global*self.using_cdvs
                else:
                    # factor=(torch.exp(torch.tensor(-0.55/temp_loss)))
                    factor=temp_loss
                    print ("global param:{}".format(factor))
                    scores=scores+scores_global*factor

            scores, ranks = torch.sort(scores, dim=0, descending=True)
            avg_ndist = torch.tensor(0).float().cuda()  # for statistics
            n_ndist = torch.tensor(0).float().cuda()  # for statistics
            # selection of negative examples
            self.nidxs = []
            for q in range(len(self.qidxs)):
                # do not use query cluster,
                # those images are potentially positive
                qcluster = self.clusters[self.qidxs[q]]
                clusters = [qcluster]
                nidxs = []
                r = 0
                while len(nidxs) < self.nnum:
                    potential = idxs2images[ranks[r, q]]
                    # take at most one image from the same cluster
                    if (not self.clusters[potential] in clusters) and (self.clusters[potential] != -1):
                        nidxs.append(potential)
                        clusters.append(self.clusters[potential])
                        avg_ndist += torch.pow(qvecs[:, q] - poolvecs[:, ranks[r, q]] + 1e-6, 2).sum(dim=0).sqrt()
                        n_ndist += 1
                    r += 1
                self.nidxs.append(nidxs)
            print('>>>> Average negative l2-distance: {:.2f}'.format(avg_ndist / n_ndist))
            print('>>>> Done')

        return (avg_ndist / n_ndist).item()  # return average negative l2-distance
