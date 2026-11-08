"""Reproducible descriptive figures generated only from saved run artifacts."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_results(output_dir):
    root=Path(output_dir); frame=pd.read_csv(root/'metrics.csv')
    names=list(frame.condition.unique());n=len(names)
    labels=[s.replace('__',' / ').replace('_',' ') for s in names]
    fig,axes=plt.subplots(1,3,figsize=(16,max(4.,.43*n+1.8)),sharey=True)
    colors=['#276b97','#d07a38','#5a8c65']
    panels=[('test_id','accuracy','ID accuracy'),('test_delay','accuracy','Unseen-delay accuracy'),
            ('test_delay','accuracy_incongruent','Unseen delay: conflict trials')]
    for ax,color,(split,metric,title) in zip(axes,colors,panels):
        for i,name in enumerate(names):
            vals=frame[(frame.condition==name)&(frame.split==split)][metric].to_numpy()*100
            offsets=np.linspace(-.12,.12,len(vals))
            ax.scatter(vals,i+offsets,s=22,color=color,alpha=.75)
            ax.plot([vals.mean()-.001,vals.mean()+.001],[i-.19,i+.19],color='black',lw=2)
        ax.axvline(50,color='#a9afb6',lw=1,ls='--');ax.set_xlim(0,100)
        ax.set_title(title,fontsize=11);ax.set_xlabel('Accuracy (%)')
        ax.grid(axis='x',alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(range(n),labels,fontsize=8);axes[0].invert_yaxis()
    fig.suptitle('Independent network runs | dots: runs; black tick: mean',fontsize=12)
    fig.tight_layout();fig.savefig(root/'behavior_summary.png',dpi=190,bbox_inches='tight')
    fig.savefig(root/'behavior_summary.pdf',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.5))
    for name in names:
        curves=[]
        for path in sorted((root/'runs'/name).glob('*/learning_curve.csv')):
            df=pd.read_csv(path);df=df.dropna(subset=['validation_bce'])
            curves.append(df.set_index('step')['validation_bce'])
        if curves:
            table=pd.concat(curves,axis=1);ax.plot(table.index,table.mean(axis=1),label=name.replace('__',' / '))
    ax.set_xlabel('Training minibatch updates');ax.set_ylabel('Validation decision BCE')
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    ax.legend(fontsize=7,bbox_to_anchor=(1.02,1),loc='upper left')
    fig.tight_layout();fig.savefig(root/'learning_curves.png',dpi=190,bbox_inches='tight')
    fig.savefig(root/'learning_curves.pdf',bbox_inches='tight');plt.close(fig)
