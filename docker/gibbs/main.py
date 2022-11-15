import numpy as np
from collections import defaultdict
import InitializationAndFirstPhase
import SecondPhasetrial
import ThirdPhase
import FourthPhase
import EpsiGreedyMultiArme
import os
import matplotlib.pyplot as plt
import FairnessTrial
np.random.seed(0)


probabilitie = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
total_times = [10**4, 5*10**4, 10**5, 2*10**5] #Time horizon

average_utility = defaultdict(list)     #create a dictionarry which assign an empty list to any unknown lement

for total_time in total_times:       # Iteration over different time horizon
    beta = total_time ** (-1. / 3.) * (np.log(total_time)) ** (1.0 / 3.0)   # precision of the expected reward for an arm
    alpha = np.sqrt(len(probabilitie))/len(probabilitie)**(2.0/3.0)*len(probabilitie)**(5.0/3.0)*total_time**(-1./3.)*(np.log(total_time))**(1.0/3.0)  #precision of the fairness function
    time_1 = defaultdict(float)
    time_2 = defaultdict(float)
    time_3 = defaultdict(float)
    time_4 = defaultdict(float)
    reward_lst = defaultdict(float)
    lambs = [0.001,0.4,0.8]
    episode = 50

    for lamb in lambs:
        for k in range(episode):

            #Phase1
            phase1 = InitializationAndFirstPhase.Environment(K=len(probabilitie), T=total_time, probability=probabilitie)
            lcb_trial1, ucb_trial1, mu_trial1, hypercube_trial1, t_trial1, update_bound_trial1, count_trial1, reward_trial1 = phase1.Explorations(lamb=lamb, bet=beta)
            print('\n the first phase gives:\n','expected reward per arm :', mu_trial1,'\n expected hypercube:', hypercube_trial1,'\n time of execution:',t_trial1,
                  '\n time of execution per arm:', count_trial1,'\n lowest lcb:',lcb_trial1, '\nreward:', reward_trial1)

            #Phase2
            phase2 = SecondPhasetrial.Phase2(mu=mu_trial1, alpha=alpha, lamb=lamb, lcb=lcb_trial1, count=count_trial1, t=t_trial1, probability=probabilitie, T=total_time,
                                             hypercube=hypercube_trial1, total_reward=reward_trial1)
            hypercube_trial2, lcb_trial2, mu_trial2, count_trial2, t_trial2, reward_trial2 = phase2.second_phase()
            print('the second phase give\n','hypercube for each arm:',hypercube_trial2, '\n expected reward per arm:', mu_trial2,'\n time of execution per arm:', count_trial2,
                  '\n time of execution:', max(t_trial2-t_trial1,0),'\n lowest lcb:',lcb_trial2, '\nreward:', reward_trial2)

            time_1[lamb] += t_trial1/(episode)
            time_2[lamb] += t_trial2/(episode)

            if max(t_trial2-t_trial1,0) + max(t_trial1,0) != 0:

                #Phase3
                phase3 = ThirdPhase.Phase3(lcb = lcb_trial2, lamb=lamb, count=count_trial2, t = t_trial2, T=total_time, mu=mu_trial2, probability=probabilitie, total_reward=reward_trial2)
                count_trial3, t_trial3, lcb_trial3, reward_trial3 = phase3.Phase3_estimation()
                print('\n The third phase lead us to:\n','\n time of execution per arm:',count_trial3,'\n time of execution:' , max(0,t_trial3-t_trial2), '\nreward:', reward_trial3)
                time_3[lamb] += max(t_trial3 - t_trial2, 0)/(episode)

                #Phase4
                phase4 = FourthPhase.Phase4(mu=mu_trial2, t=t_trial3, T=total_time, probability=probabilitie, count=count_trial3, total_reward=reward_trial3)
                count_trial4, t_trial4, mu_trial4, reward_trial4 = phase4.phase4_estimation()


                #Compute the fairness function
                softmax = FairnessTrial.softmax(coef=0.2, estimation=mu_trial4)
                estimate_softmax = softmax.softmax_estimate()
                regret_list = np.array([a*total_time for a in estimate_softmax]) - np.array([b for b in count_trial4])
                regret = lamb*np.sum(np.array([max(0,b) for b in regret_list]))
                time_4[lamb] += max(t_trial4 - t_trial3, 0) / (episode)
                reward_lst[lamb] += (reward_trial4 - regret) / (episode*total_time)
                print('The fourth phase lead us to:\n' , '\n expected reward per arm:', mu_trial4,'\n time of execution:', max(0,t_trial4-t_trial3), '\nreward:', reward_trial4 - regret)


                if k == episode-1:
                    average_utility[lamb].append(reward_lst[lamb])
                if lamb == lambs[-1] and  k == episode-1:
                    output_dir = os.path.join(os.getcwd(), 'output')
                    save_fig = True
                    for lamb in lambs:
                        plot_time = [time_2[lamb], time_3[lamb], time_4[lamb]]
                        if lamb == lambs[0]:
                            label = ['','Fairness', 'Exploration + Exploitation ']
                        else:
                            label = ['Exploration', 'Fairness', 'Exploitation']

                        # Creating explode data
                        explode = (0.1, 0.0, 0.1)
                        # Creating color parameters
                        colors = ("orange", "cyan", "green")
                        # Wedge properties
                        wp = {'linewidth': 1, 'edgecolor': "green"}
                        # Creating autocpt arguments
                        def func(pct, allvalues):
                            absolute = int(pct / 100. * np.sum(allvalues))
                            return "{:.1f}%\n({:d} g)".format(pct, absolute)
                        # Creating plot
                        fig, ax = plt.subplots(figsize=(10, 7))
                        wedges, texts, autotexts = ax.pie(plot_time,
                                                          autopct=lambda pct: func(pct, plot_time),
                                                          explode=explode,
                                                          labels=label,
                                                          shadow=True,
                                                          colors=colors,
                                                          startangle=90,
                                                          wedgeprops=wp,
                                                          textprops=dict(color="blue"))
                        # Adding legend
                        ax.legend(wedges, label,
                                  title="Phase",
                                  loc="center left",
                                  bbox_to_anchor=(1, 0, 0.5, 1))
                        plt.setp(autotexts, size=8, weight="bold")
                        ax.set_title(r' $\lambda=${} and T ={}'.format(lamb, total_time), fontsize='large')
                        if save_fig:
                            if not os.path.exists(output_dir): os.mkdir(output_dir)
                            plt.savefig(os.path.join(output_dir, 'piechart lam={}andT={}.png'.format(lamb,total_time)), bbox_inches='tight')
                        else:
                            plt.show()
                        plt.close()
                if total_time == total_times[-1] and lamb == lambs[-1] and  k == episode-1:
                    output_dir = os.path.join(os.getcwd(), 'output')
                    save_fig = True
                    for lamb in lambs[1:]:
                        plot_time_lamb = average_utility[lamb]
                        plt.plot(total_times, plot_time_lamb)
                        plt.xlabel("Horizon")
                        plt.ylabel('Average_utility')
                        plt.title(r'Softmax Fairness $\lambda=${}'.format(lamb))
                        if save_fig:
                            if not os.path.exists(output_dir): os.mkdir(output_dir)
                            plt.savefig(os.path.join(output_dir, 'average_utility lamb={}andT={}.png'.format(lamb, total_time)),
                                        bbox_inches='tight')
                        else:
                            plt.show()
                        plt.close()


            else:
                t_trial3 = 0
                t_trial4 = total_time
                time_4[lamb] += t_trial4/(episode)
                if k == episode-1 and total_time == total_times[-1]:
                    print('Normal Bandit Policy: epsilon greedy')

                    # Setting
                    probs = probabilitie  # bandit arm probabilities of success
                    N_experiments = 40000  # Number of experiments to perform
                    N_steps = 500  # number of steps (episodes)
                    eps = 0.1  # probability of random exploration (fraction)
                    save_fig = True  # save file in the same directory
                    output_dir = os.path.join(os.getcwd(), 'output')

                    # Run multi-armed bandit experiments
                    print('Running multi-armed bandits with nActions = {}, eps = {}'.format(len(probs), eps))
                    R = np.zeros((N_steps,))  # reward history sum
                    A = np.zeros((N_steps, len(probs)))  # action history sum
                    for i in range(N_experiments):
                        actions, rewards = EpsiGreedyMultiArme.experiment(probs=probs, N_episodes=N_steps, eps=eps)  # perform experiment
                        if (i + 1) % (N_experiments / 100) == 0:
                            print('[Experiment {}/{}'.format(i + 1, N_experiments) + 'n_steps = {},'.format(
                                N_steps) + "reward_avg = {}".format(np.sum(rewards) / len(rewards)))
                        R += rewards
                        for j, a in enumerate(actions):
                            A[j][a] += 1

                    # Plot reward results
                    R_avg = R / np.float64(N_experiments)
                    steps = list(np.array(range(R_avg.shape[0])) + 1)
                    print('the shape is:', R_avg.shape)
                    plt.plot(steps,R_avg, "-")
                    plt.xlabel("Horizon")
                    plt.ylabel('Average_utility')
                    plt.title(r'Average utility $\lambda=${}'.format(lamb))
                    plt.grid()
                    ax = plt.gca()
                    plt.xlim([1, N_steps])
                    if save_fig:
                        if not os.path.exists(output_dir): os.mkdir(output_dir)
                        plt.savefig(os.path.join(output_dir, 'rewardsLamb{}andTime{}.png'.format(lamb,total_time)), bbox_inches='tight')
                    else:
                        plt.show()
                    plt.close()

                    # Plot action results
                    for i in range(len(probs)):
                        A_pct = 100 * A[:, i] / N_experiments
                        steps = list(np.array(range(len(A_pct))) + 1)
                        plt.plot(steps, A_pct, "-", linewidth=4, label="Arm {} ({:.0f}%)".format(i + 1, 100 * probs[i]))
                    print('the shape of ', A_pct.shape, A_pct)
                    plt.xlabel('step')
                    plt.ylabel('Count Percentage (%)')
                    leg = plt.legend(loc='upper left', shadow=True)
                    plt.xlim([1, N_steps])
                    plt.ylim([0, 100])
                    plt.title(r'Pull arm at each step for $\lambda=${}'.format(lamb))
                    for legobj in leg.legendHandles:
                        legobj.set_linewidth(4.0)
                    if save_fig:
                        if not os.path.exists(output_dir): os.mkdir(output_dir)
                        plt.savefig(os.path.join(output_dir, 'actions2Lamb{}andTime{}.png'.format(lamb,total_time)))
                    else:
                        plt.show()
                    plt.close()


        #if max(t_trial2-t_trial1,0)


#print(reward_lst, time_1, time_2, time_3, time_4)
