# Make a paired circles and discs toy dataset for multimodal encoding
import os

import numpy as np
from PIL import Image
import torch
from sklearn.model_selection import train_test_split

unbalanced = True # case where we don't have one for one correspondance (q(z|x) is not a dirac)
dataset_size = 10000
size_image = 32
min_rayon, max_rayon = 0.1, 0.9
circle_thickness = 0.1
n_repeat = 10
output_path = '../data/unbalanced/circles_and_discs' if unbalanced else '../data/circles_and_discs'
if not os.path.exists(output_path):
    os.mkdir(output_path)

rayons = np.linspace(min_rayon,max_rayon,dataset_size)
x = np.linspace(-1,1,size_image)

circles = []
discs = []

for i, r_disc in enumerate(rayons):
    for _ in range(n_repeat):
        X,Y = np.meshgrid(x,x)

        img_disc = X**2 + Y**2 <= r_disc**2
        r_cercle = r_disc if not unbalanced else np.random.uniform(min_rayon,r_disc)
        print(r_disc, r_cercle)
        img_circle = (X**2 + Y**2 <= (r_cercle + circle_thickness/2)**2)*(X**2 + Y**2 >= (r_cercle - circle_thickness/2)**2)

        circles.append(img_circle)
        discs.append(img_disc)



# Visualize some examples
output_examples = output_path + '/examples'
if not os.path.exists(output_examples):
    os.mkdir(output_examples)

for i in np.linspace(0,dataset_size*n_repeat-1, 100):
    print(i)
    i = int(i)
    img = Image.fromarray(np.concatenate([circles[i], discs[i]]))
    img.save(output_examples + f'/example_{i}.png')



# Save in pytorch format
circles = torch.unsqueeze(torch.FloatTensor(circles), 1)
discs = torch.unsqueeze(torch.FloatTensor(discs), 1)

# Select some for training and testing
c_train, c_test, d_train, d_test = train_test_split(circles,discs, test_size=0.3)

torch.save(c_train, output_path + '/circles_train.pt')
torch.save(c_test, output_path + '/circles_test.pt')
torch.save(d_train, output_path + '/discs_train.pt')
torch.save(d_test, output_path + '/discs_test.pt')

print(c_train.shape, c_test.shape)




