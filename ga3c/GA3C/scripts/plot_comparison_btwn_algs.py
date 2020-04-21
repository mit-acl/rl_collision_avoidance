import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm

plt.rcParams["text.usetex"] = True
plt.rcParams["font.size"] = 20
plt.rcParams['font.family'] = 'serif'

def markers(x):
    marker_list = [".","o","v","^","s","p","+","*","x","D"]
    x = x%len(marker_list)
    return marker_list[x]
def colors(x, name="tab20c"):
    return cm.get_cmap(plt.get_cmap(name))(x)

algs = [
    {
        'name': 'RVO',
        'color': colors(16),
        'marker': markers(0),
    },
        {
        'name': 'CADRL',
        'color': colors(0),
        'marker': markers(1),
    },
    {
        'name': 'GA3C-CADRL-4-WS-4',
        'color': colors(4),
        'marker': markers(2),
        'seeds': ['1','2','3','4','5'],
    },
    {
        'name': 'GA3C-CADRL-4-WS-6',
        'color': colors(5),
        'marker': markers(2),
        'seeds': ['1','2','3','4'],
        'skip_above': 4,
    },
    {
        'name': 'GA3C-CADRL-4-WS-8',
        'color': colors(6),
        'marker': markers(2),
        'seeds': ['1','2','3','4'],
        'skip_above': 4,
    },
    {
        'name': 'GA3C-CADRL-4-LSTM',
        'color': colors(9),
        'marker': markers(3),
        'seeds': ['1','2','3','4','5'],
    },
    {
        'name': 'GA3C-CADRL-10-WS-4',
        'color': colors(12),
        'marker': markers(4),
        'seeds': ['1','2','3','4','5'],
    },
    {
        'name': 'GA3C-CADRL-10-WS-6',
        'color': colors(13),
        'marker': markers(4),
        'seeds': ['1','2','3','4'],
    },
    {
        'name': 'GA3C-CADRL-10-WS-8',
        'color': colors(14),
        'marker': markers(4),
        'seeds': ['1','2','3','4'],
    },
    {
        'name': 'GA3C-CADRL-10-LSTM',
        'color': colors(8),
        'marker': markers(5),
        'seeds': ['1','2','3','4','5'],
    },
]

dir = "/mnt/ubuntu_extra_ssd3/code/rl_collision_avoidance/gym-collision-avoidance/gym_collision_avoidance/experiments/src"

column_order = [alg['name'] for alg in algs]
stats = [
    {
        'name': 'pct_failures',
        'ylabel': '\% Failures'
    },
    {
        'name': 'pct_collisions',
        'ylabel': '\% Collisions'
    },
    {
        'name': 'pct_stuck',
        'ylabel': '\% Stuck'
    },
    {
        'name': 'extra_time_to_goal_50',
        'ylabel': r'Extra Time, $\bar{t}_g^e\ (50^{th}$ \%tile)'
    },
    {
        'name': 'extra_time_to_goal_75',
        'ylabel': r'Extra Time, $\bar{t}_g^e\ (75^{th}$ \%tile)'
    },
    {
        'name': 'extra_time_to_goal_90',
        'ylabel': r'Extra Time, $\bar{t}_g^e\ (90^{th}$ \%tile)'
    }
]

def plot(small=True, num_agents_to_test=[]):
    df = pd.DataFrame()

    for num_agents in num_agents_to_test:
        for policy in algs:
            if 'skip_above' in policy.keys():
                if num_agents > policy['skip_above']:
                    continue
            if 'seeds' in policy.keys():
                seeds = ['-'+s for s in policy['seeds']]
            else:
                seeds = ['']
            for seed in seeds:
                pol = policy['name']+seed
                log_filename = '{dir}/../results/full_test_suites/{num_agents}_agents/stats/stats_{policy}.p'.format(dir=dir, num_agents=num_agents, policy=pol) 
                try:
                    df_ = pd.read_pickle(log_filename)
                except:
                    continue
                num_test_cases = df_.shape[0]
                
                successful_cases = df_.all_at_goal == 1
                if not np.any(successful_cases):
                    extra_time_to_goal = extra_time_to_goal_50 = extra_time_to_goal_75 = extra_time_to_goal_90 = 0
                else:
                    # Compute mean of each episode, then compute percentiles of means
                    extra_time_to_goal = np.mean(np.stack(df_.extra_time_to_goal[successful_cases].to_numpy()), axis=1)
                    extra_time_to_goal_50, extra_time_to_goal_75, extra_time_to_goal_90 = np.percentile(extra_time_to_goal,[50,75,90])
                pct_collisions = 100.*np.sum(df_.collision)/num_test_cases
                pct_stuck = 100.*np.sum(df_.any_stuck)/num_test_cases
                df = df.append({
                    'extra_time_to_goal': extra_time_to_goal,
                    'extra_time_to_goal_50': extra_time_to_goal_50,
                    'extra_time_to_goal_75': extra_time_to_goal_75,
                    'extra_time_to_goal_90': extra_time_to_goal_90,
                    'pct_failures': pct_collisions+pct_stuck,
                    'pct_collisions': pct_collisions,
                    'pct_stuck': pct_stuck,
                    'num_agents': num_agents,
                    'policy_id': policy['name'],
                    'seed': seed,
                    'pol': pol
                }, ignore_index=True)
    df.num_agents = df.num_agents.astype(int)
    # print(df)

    df_failures = pd.pivot_table(df, index=['num_agents', 'policy_id'],
                   aggfunc={
                       'pct_failures': ['mean', 'std'],
                       'pct_collisions': ['mean', 'std'],
                       'pct_stuck': ['mean', 'std'],
                       'extra_time_to_goal_50': ['mean', 'std'],
                       'extra_time_to_goal_75': ['mean', 'std'],
                       'extra_time_to_goal_90': ['mean', 'std'],
                   }).reset_index(level=[0,1])
    # print(df_failures)

    fig, axes = plt.subplots(2,3, figsize=(24,8))
    aggfunc = {}
    for stat in stats:
        aggfunc[stat['name']] = ['mean', 'std', list]
    df_failures = pd.pivot_table(df, index=['num_agents', 'policy_id'],
                   aggfunc=aggfunc).reset_index(level=[0,1])
    for i, ax in enumerate(axes.ravel()):
        stat = stats[i]['name']
        
        li = df_failures.pivot(index='num_agents', columns='policy_id', values=(stat,'list')).reindex(column_order, axis=1)
        mean = df_failures.pivot(index='num_agents', columns='policy_id', values=(stat,'mean')).reindex(column_order, axis=1)
        std = df_failures.pivot(index='num_agents', columns='policy_id', values=(stat,'std')).reindex(column_order, axis=1).fillna(0)
        mean.plot(ax=ax, kind='bar', yerr=std, rot=0, width=0.9,
                 color=[alg['color'] for alg in algs])
        ax.set_ylabel(stats[i]['ylabel'])
        ax.set_xlabel("Number of Agents")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_ylim(bottom=0)
        
        AxesDecorator(ax, ticks=ax.get_xticks())
        ax.get_legend().remove()
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='center left')

    plt.tight_layout()
    if small:
        fig.savefig('/home/mfe/test_time_bars_n_small.pdf')
    else:
        fig.savefig('/home/mfe/test_time_bars_n_large.pdf')


class AxesDecorator():
    def __init__(self, ax, size="5%", pad=0.05, ticks=[1,2,3], spacing=0.05,
                 color="k"):
        self.divider= make_axes_locatable(ax)
        self.ax = self.divider.new_vertical(size=size, pad=pad, sharex=ax, pack_start=True)
        ax.figure.add_axes(self.ax)
        self.ticks=np.array(ticks)
        self.d = np.mean(np.diff(ticks))
        self.spacing = spacing
        self.get_curve()
        self.color=color
        for x0 in ticks:
            self.plot_curve(x0)
        self.ax.set_yticks([])
        plt.setp(ax.get_xticklabels(), visible=False)
        self.ax.tick_params(axis='x', which=u'both',length=0)
        ax.tick_params(axis='x', which=u'both',length=0)
        for direction in ["left", "right", "bottom", "top"]:
            self.ax.spines[direction].set_visible(False)
        self.ax.set_xlabel(ax.get_xlabel())
        ax.set_xlabel("")
        self.ax.set_xticks(self.ticks)

    def plot_curve(self, x0):
        x = np.linspace(x0-self.d/2.*(1-self.spacing),x0+self.d/2.*(1-self.spacing), 50 )
        self.ax.plot(x, self.curve, c=self.color)

    def get_curve(self):
        lx = np.linspace(-np.pi/2.+0.05, np.pi/2.-0.05, 25)
        tan = np.tan(lx)*10
        self.curve = np.hstack((tan[::-1],tan))
        return self.curve

if __name__ == '__main__':
    plot(small=True, num_agents_to_test = [2,3,4])
    plot(small=False, num_agents_to_test = [6,8,10])
