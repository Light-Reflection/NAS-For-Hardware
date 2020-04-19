import dmmo as dm
import torch
model = dm.mobilenet_supernet(channels = [(32,1),(64,1), [(128,2),(256,2),(512,2),(1024,2)], (320,1), (1280,1)],
                    layers = 15, num_of_classes = 10,name = 'testing', division=2, 
                    search_direction = [True,True,True,False],
                    constrain = 1000000000,
                    num_of_ops = 2,

                   ).cuda()

ren = [2, 2, 6, 2]
cen = [2] * 14
oen = [0] * 12
# cen = [1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2]
# cen = [2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1,1]
# oen = [0] * 12
#oen = [0,0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# oen = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
model = model.dispatch(ren, cen, oen).cpu()
print(model)
dm.test_model('mobilenet-c', model)

#print(model)



