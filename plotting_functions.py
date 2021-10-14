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

figure_dir = "./Figures/"

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
        

def plot_fr(results,smooth=False):
    
    # Plot Population firing rates
    results_local = results['popfreqs'].copy()
    if smooth==True:
        results_local = smoothen_fr(results_local)
    
    results_local_melt = results_local.melt("Time (ms)")
    
    
    results_local_melt["nuclei"] = [ x.split('_')[0]  for x in results_local_melt["variable"]]
    results_local_melt["channel"] = [ x.split('_')[1]  for x in results_local_melt["variable"]]
    #print(results_local_melt)
    results_local_melt = results_local_melt.rename(columns={"value":"firing_rate"})
    col_order = ["D1STR", "GPeP", "GPi","D2STR", "STNE", "Th", "Cx","CxI","FSI"] # To ease comparison with reference Figure 
                 

    g1 = sns.relplot(x="Time (ms)", y ="firing_rate", hue="channel",col="nuclei",data=results_local_melt,col_wrap=3,kind="line",facet_kws={'sharey': False, 'sharex': True},col_order=col_order)
    if smooth == True:
        g1.fig.savefig(figure_dir+'ActualFR_smooth.png', dpi=400)
    else:
        g1.fig.savefig(figure_dir+'ActualFR.png', dpi=400)
    
def plot_reward_and_Q_df(t_epochs,Q_df,dat_tabs):
    Q_df_local = Q_df.copy()
    Q_df_local = Q_df_local.reset_index()
    Q_df_local.index-=1
    Q_df_local["Trials"] = Q_df_local.index
    Q_df_local = Q_df_local.melt("Trials")
    Q_df_local = Q_df_local.loc[Q_df_local["variable"]!= "index"]
    Q_df_local = Q_df_local.reset_index()
    Q_df_local["data_type"] = "Q_df"
    
    rew_df = t_epochs.copy()
    rew_df["Trials"] = t_epochs.index
    rew_df = rew_df.melt("Trials")
    rew_df["data_type"] = "reward_df"
    
    chosen_action = pd.DataFrame(dat_tabs["decision"].copy())
    chosen_action["Trials"] = chosen_action.index
    chosen_action = chosen_action.rename(columns={"decision":"variable"})
    chosen_action["value"] = chosen_action.groupby("variable").ngroup()
    chosen_action["data_type"] = "chosen action"
    print(chosen_action)
    
    final_data = Q_df_local.append(rew_df)
    final_data = final_data.append(chosen_action)

    g1 = sns.catplot(x="Trials",y="value",hue="variable",col="data_type",data=final_data,kind='point')
    for x in g1.axes[0]:
        x.set_xticklabels(x.get_xticklabels(),fontsize=10,fontweight='bold')
    xlim = g1.axes[0][0].get_xlim()
    g1.axes[0][0].hlines(y=0.5,xmin=xlim[0],xmax=xlim[1],ls='dashed',color='k',lw=2.0)
    
    g1.fig.savefig(figure_dir+"Reward_and_Q_df.png")