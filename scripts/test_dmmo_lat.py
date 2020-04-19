import dmmo as dm
import torch
model = dm.supernet(channels = [(32,1), (16,1), [(24,2),(32,2),(64,2),(96,1),(160,2)], (320,1), (1280,1)],
                    layers = 20,num_of_classes = 10,name = 'testing', division=2, 
                    search_direction = [True,True,True,False],
                    constrain = 1000000000,
                   ).cuda()

ren = [2, 3, 4, 3, 3]
    # cen = [1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2]
cen = [2]*19
oen = [0]*15
# oen = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
model = model.dispatch(ren, cen, oen).cpu()
dm.test_model('modelv2-cpu', model)

#print(model)



