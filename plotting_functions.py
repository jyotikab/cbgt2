import cbgt as cbgt
from frontendhelpers import *
from tracetype import *
import init_params as par
import popconstruct as popconstruct
import qvalues as qval
import generateepochs as gen
from agentmatrixinit import *
from agent_timestep import timestep_mutator, multitimestep_mutator
import pipeline_creation as pl_creat
import seaborn as sns
import matplotlib.pyplot as plt
import pylab as pl
import plotting_helper_functions as plt_help

figure_dir = "./Figures/"
data_dir = "./Data/"

def rename_columns(results,smooth=False):
    
    results['popdata']['newname'] = results['popdata']['name']+'_'+results['popdata']['action']
    new_names = dict()
    for i in results['popdata'].index[:-2]:
        temp = untrace(results['popdata']['newname'].iloc[i])
        #print(type(temp))
        if 'LIP' in temp:
            temp1 = "Cx_"+temp.split('_')[1]
            temp = temp1
        new_names[i] = temp
    new_names[i+1]='FSI_common'
    new_names[i+2]='CxI_common'
    results['popfreqs'] = results['popfreqs'].rename(columns=new_names)
    
    return results


def smoothen_fr(results,win_len=50):
    
    win = np.ones(win_len)/float(win_len)
        
    for k in list(results.keys()):
        if "Time" in k:
            continue
        results[k] = np.convolve(results[k],win,mode='same')
                
    return results
        
def plot_fr(results,seed):
    
    # Plot Population firing rates
    col_order = ["D1STR", "GPeP", "GPi","D2STR", "STNE", "Th", "Cx","CxI","FSI"] # To ease comparison with reference Figure 
                 
    for i in np.arange(len(results)):
        g1 = sns.relplot(x="Time (ms)", y ="firing_rate", hue="channel",col="nuclei",data=results[i],col_wrap=3,kind="line",facet_kws={'sharey': False, 'sharex': True},col_order=col_order)
        g1.fig.savefig(figure_dir+'ActualFR_'+str(seed)+"_"+str(i)+".png", dpi=400)
        
        
def plot_reward_Q_df(final_data):

    colors = list(sns.color_palette())
    
    var = np.unique(final_data[0]["variable"])
    col_dict = dict()
    for i,v in enumerate(var):
        col_dict[v] = colors[i]
    
    for i in np.arange(len(final_data)):
                       
        g1 = sns.catplot(x="Trials",y="value",hue="variable",col="data_type",data=final_data[i],kind='point',col_wrap=2,sharey=False,palette=col_dict)
        for x in g1.axes:
            x.set_xticklabels(x.get_xticklabels(),fontsize=10,fontweight='bold')
            if np.max(final_data[i]["Trials"]) > 10:
                x.set_xticklabels([])
       
        xlim = g1.axes[0].get_xlim()
        g1.fig.savefig(figure_dir+"Reward_and_Q_df_"+final_data[i]["seed"].values[0]+".png")
    
def performance_all(performance=[],rt_dist=[]):
    if len(performance) == 0 and len(rt_dist) == 0:
        print("Pooling data")
        plt_help.pool_data()
        performance = pd.read_csv(data_dir+"performance_all.csv")
        rt_dist = pd.read_csv(data_dir+"rt_distribution_all.csv")
        post_fix = "all"
    else:
        post_fix = str(performance["seed"][0]).split('_')[0]
    g1 = sns.catplot(x="block",y="%_rewarded_actions",hue="actions",data=performance,col="conflict",kind="bar")
    g1.fig.savefig(figure_dir+"Performance_rewarded_actions_all.png")
    g4 = sns.catplot(x="block",y="%_action",hue="actions",data=performance,col="conflict",kind="bar")
    g4.fig.savefig(figure_dir+"Performance_actions_"+post_fix+".png")
    
    rt_dist = rt_dist.reset_index()
    pl.figure()
    hist = sns.histplot(x="decisiondurationplusdelay",data=rt_dist,hue="conflict",kde=True,palette="deep",stat="density")
    hist.figure.savefig(figure_dir+"RT_distribution_"+post_fix+".png")


    
    
    