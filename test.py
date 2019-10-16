import dmmo as dm
model = dm.supernet(num_of_classes = 10,name = 'testing').cuda()


model.fit(model= model,
         data_root = '/home/dm/data/CIFAR',
         epoch = 1)
