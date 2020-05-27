import numpy as np
from fastai.vision import *
from fastai.callbacks.hooks import *
import scipy.ndimage
import gc
import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


# Helper functions

def np2tensor(image,dtype):
    "Convert np.array (sz,sz,3) to tensor (1,3,sz,sz), imagenet normalized"

    a = np.asarray(image)
    if a.ndim==2 : a = np.expand_dims(a,2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    
    # #Imagenet norm
    # mean=np.array([0.485, 0.456, 0.406])[...,np.newaxis,np.newaxis]
    # std = np.array([0.229, 0.224, 0.225])[...,np.newaxis,np.newaxis]
    #cifar norm
    mean=np.array([0.4914, 0.4822, 0.4465])[...,np.newaxis,np.newaxis]
    std = np.array([0.247, 0.243, 0.261])[...,np.newaxis,np.newaxis]
    a = (a-mean)/std
    a = np.expand_dims(a,0)
    return torch.from_numpy(a.astype(dtype, copy=False) )

def tensor2np(img_tensor):
    "Convert tensor (1,3,sz,sz) back to np.array (sz,sz,3), imagenet DEnormalized"
    a = np.squeeze(to_np(img_tensor))
    
    # # Imagenet norm
    # mean=np.array([0.485, 0.456, 0.406])[...,np.newaxis,np.newaxis]
    # std = np.array([0.229, 0.224, 0.225])[...,np.newaxis,np.newaxis]
    #cifar norm
    mean=np.array([0.4914, 0.4822, 0.4465])[...,np.newaxis,np.newaxis]
    std = np.array([0.247, 0.243, 0.261])[...,np.newaxis,np.newaxis]
    a = a*std + mean
    return np.transpose(a, (1,2,0))

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output
    def close(self):
        self.hook.remove()



class FilterVisualizer():
    def __init__(self, model):
        self.model = model

    def visualize(self, layer, filter, sz = 56, upscaling_steps=12, upscaling_factor=1.2, lr=0.1, opt_steps=20, blur=None, print_losses=False):
        
        img = (np.random.random((sz,sz, 3)) * 20 + 128.)/255 # value b/t 0 and 1
        # img = np.uint8(np.random.uniform(150, 180, (sz, sz, 3)))        
        activations = SaveFeatures(layer)  # register hook

        for i in range(upscaling_steps):  
            # convert np to tensor + channel first + new axis, and apply [insert dataset] norm
            img_tensor = np2tensor(img,np.float32)
            img_tensor = img_tensor.cuda()
            img_tensor.requires_grad_();
            if not img_tensor.grad is None:
                img_tensor.grad.zero_(); 
            
            optimizer = torch.optim.Adam([img_tensor], lr=lr, weight_decay=1e-6)
            
            if i > upscaling_steps/2:
                opt_steps_ = int(opt_steps*1.3)
            else:
                opt_steps_ = opt_steps
            # opt_steps_ = opt_steps
            for n in range(opt_steps_):  # optimize pixel values for opt_steps times
                optimizer.zero_grad()
                _=self.model(img_tensor)
                loss = -1*activations.features[0, filter].mean()
                if print_losses:
                    if i%3==0 and n%5==0:
                        print(f'{i} - {n} - {float(loss)}')
                loss.backward()
                optimizer.step()
            
            # convert tensor back to np
            img = tensor2np(img_tensor)
            self.output = img
            sz = int(upscaling_factor * sz)  # calculate new image size
#             print(f'Upscale img to: {sz}')
            img = cv2.resize(img, (sz, sz), interpolation = cv2.INTER_CUBIC)  # scale image up
            if blur is not None: img = cv2.blur(img,(blur,blur))  # blur image to reduce high frequency patterns
                
        activations.close()
        return np.clip(self.output, 0, 1)

    def get_transformed_img(self,img,sz):
      '''
      Scale up/down img to sz. Channel last (same as input)
      image: np.array [sz,sz,3], already divided by 255"    
      '''
      return cv2.resize(img, (sz, sz), interpolation = cv2.INTER_CUBIC)

    def most_activated(self, img, layer):
      '''
      image: np.array [sz,sz,3], already divided by 255"    
      '''
      img = cv2.resize(img, (224,224), interpolation = cv2.INTER_CUBIC)
      activations = SaveFeatures(layer)
      img_tensor = np2tensor(img,np.float32)
      img_tensor = img_tensor.cuda()
      
      _=self.model(img_tensor)
      mean_act = [np.squeeze(to_np(activations.features[0,i].mean())) for i in range(activations.features.shape[1])]
      activations.close()
      return mean_act

# To use below functions, need to define a FilterVisuzliazer class and pass as an input

def activations_and_reconstructions(img,FV,fmap_layer,
                                    top_num=4,init_size=56,
                                    upscaling_steps=12, upscaling_factor=1.2,
                                    opt_steps=20, blur=5,lr=1e-1,
                                    print_losses=False,
                                    n_cols=3, cell_size=4,
                                    layer_name='',
                                    save_fig=False,
                                    album_hash=None):
    
    mean_acts = FV.most_activated(img,layer = fmap_layer)

    most_act_fmaps = sorted(range(len(mean_acts)), key=lambda i: mean_acts[i])[-top_num:][::-1]

    imgs = []
    for filter in most_act_fmaps:
        imgs.append(FV.visualize(fmap_layer, filter, upscaling_steps=upscaling_steps, 
                                 sz = init_size,
                                 upscaling_factor=upscaling_factor, 
                                 opt_steps=opt_steps, blur=blur,
                                 lr=lr,print_losses=False))
    transformed_img = FV.get_transformed_img(img,224)
    
    plot_activations_and_reconstructions(imgs,mean_acts,
                                         most_act_fmaps,transformed_img,
                                         n_cols=n_cols,cell_size=cell_size,
                                         layer_name=layer_name,
                                         save_fig=save_fig,
                                         album_hash=album_hash)

def custom_activations_and_reconstructions(img,FV,fmap_layer,
                                    feat_index,init_size=56,
                                    upscaling_steps=12, upscaling_factor=1.2,
                                    opt_steps=20, blur=5,lr=1e-1,
                                    print_losses=False,
                                    n_cols=3, cell_size=4,
                                    layer_name='',
                                    save_fig=False,
                                    album_hash=None,
                                    ):
    """
    Used to input a custom list of filters for a given layer
    """
    newlist = []
    act_layers = FV.most_activated(img,layer = fmap_layer)

    # COnstruct list to use for indexing custom featmap
    for i in feat_index:
      newlist.append(i)

    imgs = []
    for filter in newlist:
        imgs.append(FV.visualize(fmap_layer, filter, upscaling_steps=upscaling_steps, 
                                 sz = init_size,
                                 upscaling_factor=upscaling_factor, 
                                 opt_steps=opt_steps, blur=blur,
                                 lr=lr,print_losses=False))
    transformed_img = FV.get_transformed_img(img,224)
    
    plot_activations_and_reconstructions(imgs,mean_acts,
                                         newlist,transformed_img,
                                         n_cols=n_cols,cell_size=cell_size,
                                         layer_name=layer_name,
                                         save_fig=save_fig,
                                         album_hash=album_hash)

def plot_activations_and_reconstructions(imgs,activations,filters,
                                         transformed_img,n_cols=3,
                                         cell_size=4,layer_name='',
                                         save_fig=False,album_hash=None):
    n_rows = math.ceil((len(imgs)+1)/n_cols)

    fig = plt.figure(figsize=(cell_size*n_cols,cell_size*n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols)
    tr_im_ax = plt.subplot(gs[0,0])
    tr_im_ax.grid(False)
    tr_im_ax.get_xaxis().set_visible(False)
    tr_im_ax.get_yaxis().set_visible(False)
    tr_im_ax.imshow(transformed_img)
    tr_im_ax.set_title('Image')
    
    act_ax = plt.subplot(gs[0, 1:])
    
    
    act = act_ax.plot(np.clip(activations,0.,None),linewidth=2.)
    for el in filters:
        act_ax.axvline(x=el, color='red', linestyle='--',alpha=0.4)
    act_ax.set_xlim(0,len(activations));
    act_ax.set_ylabel(f"mean activation");
    if layer_name == '':
        act_ax.set_title('Mean Activations')
    else:
        act_ax.set_title(f'{layer_name}')
    act_ax.set_facecolor('white')
    
    fmap_axes = []
    for r in range(1,n_rows):
        for c in range(n_cols):
            fmap_axes.append(plt.subplot(gs[r, c]))
            
    for i,ax in enumerate(fmap_axes):
        ax.grid(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i>=len(filters):
            pass

        ax.set_title(f'fmap {filters[i]}')

        ax.imshow(imgs[i])
    plt.tight_layout()
    save_name = layer_name.lower().replace(' ','_')
    if save_fig:
        plt.savefig(f'{save_name}.png')
        plt.close()
    else:
        plt.show()