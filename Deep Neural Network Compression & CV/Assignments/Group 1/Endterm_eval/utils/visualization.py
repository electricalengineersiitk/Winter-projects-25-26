# visualization utilities for the compression pipeline
# all the plotting functions for weight distributions, sparsity, training curves etc.
# using matplotlib just like in the assignments

import matplotlib.pyplot as plt
import numpy as np
import torch

# FIX: Changed relative import to absolute import to prevent ImportError
from compression.linear import CompressedLinear
from compression.conv2d import CompressedConv2d

def plot_weight_distribution(model,title="Weight Distribution",
                             save_path=None):
    # plot histograms of weight values for each layer
    # shows active weights (after pruning) with how many were pruned
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    fig.suptitle(title,fontsize=14,fontweight='bold')

    layer_idx=0
    for name,module in model.named_modules():
        if isinstance(module, (CompressedLinear, CompressedConv2d, torch.nn.Linear, torch.nn.Conv2d)):
            if layer_idx >= 4:
                break
            ax=axes[layer_idx//2][layer_idx%2]

            weights=module.weight.data.cpu().numpy().flatten()

            if isinstance(module, (CompressedLinear, CompressedConv2d)):
                mask=module.mask.data.cpu().numpy().flatten()
                active=weights[mask>0]
                ax.hist(active,bins=50,alpha=0.7,color='#2196F3',
                        edgecolor='black',linewidth=0.5,
                        label=f'Active ({len(active)})')
                pruned_count=(mask==0).sum()
                ax.set_title(f'Layer {layer_idx}: {name}\n'
                             f'Pruned: {pruned_count}/{len(weights)} '
                             f'({100*pruned_count/len(weights):.1f}%)',
                             fontsize=10)
            else:
                ax.hist(weights,bins=50,alpha=0.7,color='#4CAF50',
                        edgecolor='black',linewidth=0.5)
                ax.set_title(f'Layer {layer_idx}: {name}',fontsize=10)

            ax.set_xlabel('Weight Value',fontsize=9)
            ax.set_ylabel('Count',fontsize=9)
            ax.axvline(x=0,color='red',linestyle='--',alpha=0.5) # mark zero
            layer_idx+=1

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path,dpi=150,bbox_inches='tight')
    # plt.show()


def plot_sparsity(sparsity_dict,save_path=None):
    # horizontal bar chart showing sparsity per layer
    fig,ax=plt.subplots(figsize=(8,5))
    layers=list(sparsity_dict.keys())
    sparsities=[sparsity_dict[l]*100 for l in layers]

    bars=ax.barh(layers,sparsities,color='#ff7043',edgecolor='black')
    ax.set_xlabel('Sparsity (%)',fontsize=12)
    ax.set_title('Weight Sparsity per Layer',fontsize=14,fontweight='bold')
    ax.set_xlim(0,100)

    # add the percentage labels next to each bar
    for bar,val in zip(bars,sparsities):
        ax.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,
                f'{val:.1f}%',va='center',fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path,dpi=150,bbox_inches='tight')
    # plt.show()


def plot_training_curves(history,title="Training Curves",save_path=None):
    # plot loss and accuracy curves over epochs
    # similar to the 3-panel plot in assignment 6 but split into 2 panels
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
    fig.suptitle(title,fontsize=14,fontweight='bold')

    epochs=range(1,len(history['train_loss'])+1)

    # loss curves (like the loss plot in assignment 6)
    ax1.plot(epochs,history['train_loss'],'b-',label='Train',linewidth=2)
    ax1.plot(epochs,history['test_loss'],'r--',label='Test',linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True,alpha=0.3)

    # accuracy curves
    ax2.plot(epochs,history['train_acc'],'b-',label='Train',linewidth=2)
    ax2.plot(epochs,history['test_acc'],'r--',label='Test',linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True,alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path,dpi=150,bbox_inches='tight')
    # plt.show()


def plot_compression_summary(sizes_dict,accuracies_dict,save_path=None):
    # side by side comparison of model sizes and accuracies at each compression stage
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5))
    fig.suptitle('Deep Compression Pipeline - Summary',
                 fontsize=14,fontweight='bold')

    stages=list(sizes_dict.keys())
    sizes_kb=[sizes_dict[s]/1024 for s in stages]
    accs=[accuracies_dict[s] for s in stages]

    # model size comparison
    colors=['#4CAF50','#2196F3','#FF9800','#9C27B0']
    bars=ax1.bar(stages,sizes_kb,color=colors[:len(stages)],
                 edgecolor='black')
    ax1.set_ylabel('Model Size (KB)',fontsize=12)
    ax1.set_title('Model Size at Each Stage',fontsize=12,fontweight='bold')
    for bar,val in zip(bars,sizes_kb):
        ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
                 f'{val:.1f} KB',ha='center',fontsize=9)

    # accuracy comparison
    bars2=ax2.bar(stages,accs,color=colors[:len(stages)],
                  edgecolor='black')
    ax2.set_ylabel('Accuracy (%)',fontsize=12)
    ax2.set_title('Accuracy at Each Stage',fontsize=12,fontweight='bold')
    ax2.set_ylim(min(accs)-5,max(accs)+2)
    for bar,val in zip(bars2,accs):
        ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,
                 f'{val:.2f}%',ha='center',fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path,dpi=150,bbox_inches='tight')
    # plt.show()