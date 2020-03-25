import numpy as np
import pickle

num_agents = 4
cadrl = pickle.load(open('logs/results/%s_agents/CADRL.p'%(num_agents),'rb'))
a3c1 = pickle.load(open('logs/results/%s_agents/A3C1.p'%(num_agents),'rb'))
a3c2 = pickle.load(open('logs/results/%s_agents/A3C2.p'%(num_agents),'rb'))

algs = ['A3C1','A3C2','CADRL']
stats = {}
stats['A3C1'] = a3c1['A3C']
stats['A3C2'] = a3c2['A3C']
stats['CADRL'] = cadrl['CADRL']

print '\n\n\n#######################'
test_cases = 500
non_collision_inds = np.intersect1d(np.intersect1d(stats['A3C1']['non_collision_inds'], stats['CADRL']['non_collision_inds']),stats['A3C2']['non_collision_inds'])
all_at_goal_inds = np.intersect1d(np.intersect1d(stats['A3C1']['all_at_goal_inds'], stats['CADRL']['all_at_goal_inds']),stats['A3C2']['all_at_goal_inds'])
no_funny_business_inds = np.intersect1d(non_collision_inds, all_at_goal_inds)
for alg in algs:
    print "Algorithm: %s" %alg
    num_collisions = test_cases-len(stats[alg]['non_collision_inds'])
    num_stuck = len(stats[alg]['stuck_inds'])
    print "Total # test cases with collision: %i of %i (%.2f%%)" %(num_collisions,test_cases,(100.0*num_collisions/(test_cases)))
    print "Total # test cases where agent got stuck: %i of %i (%.2f%%)" %(num_stuck,test_cases,(100.0*num_stuck/(test_cases)))
    # time_to_goal_sum = 0.0
    mean_extra_time_to_goal_list = []
    for ind in no_funny_business_inds:
        # time_to_goal_sum += stats[alg][ind]['total_time_to_goal']
        mean_extra_time_to_goal_list.append(stats[alg][ind]['mean_extra_time_to_goal'])
    print "%s: mean extra time to goal (non-collision/non-stuck cases):" %(alg)
    print np.percentile(np.array(mean_extra_time_to_goal_list),[50,75,90])
print '#######################'