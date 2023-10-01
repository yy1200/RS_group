
import math
import random
import numpy as np
import os
import sys
from tqdm import tqdm
# sys.path.append('..')

from collections import namedtuple
import argparse
from itertools import count, chain
import torch
from utils import *
from agent import Agent
#TODO select env
from RL.env_multi_choice_question import MultiChoiceRecommendEnv
from RL.RL_evaluate import dqn_evaluate
from multi_interest import GraphEncoder
import time
import warnings
from construct_graph import get_graph
from memory import *


pid = os.getpid()
print('pid : ',pid)
warnings.filterwarnings("ignore")
EnvDict = {
    LAST_FM_STAR: MultiChoiceRecommendEnv,
    YELP_STAR: MultiChoiceRecommendEnv,
    BOOK:MultiChoiceRecommendEnv,
    MOVIE:MultiChoiceRecommendEnv,
    AMAZON:MultiChoiceRecommendEnv,
    ML1M:MultiChoiceRecommendEnv
    }
FeatureDict = {
    LAST_FM_STAR: 'feature',
    YELP_STAR: 'feature',
    BOOK:'feature',
    MOVIE:'feature',
    AMAZON:'feature',
    ML1M:'feature'
}

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'next_cand'))

def train(args, kg, filename):

    SR5_best, SR10_best, SR15_best, AvgT_best, Rank_best, NDCG_best, P_best, R_best, reward_best=0.,0.,0.,15.,0.,0.,0.,0.,0.
    env = EnvDict[args.data_name](kg, args.data_name, args.embed, seed=args.seed, max_turn=args.max_turn, cand_num=args.cand_num, cand_item_num=args.cand_item_num,
                       attr_num=args.attr_num, mode='train', ask_num=args.ask_num, entropy_way=args.entropy_method, fm_epoch=args.fm_epoch,choice_num=args.choice_num)
    set_random_seed(args.seed)
    G=get_graph(env.user_length,env.item_length,env.feature_length,env.group_length,args.data_name)
    memory = ReplayMemoryPER(args.memory_size) #50000
    embed = torch.FloatTensor(np.concatenate((env.ui_embeds, env.feature_emb, np.zeros((1,env.ui_embeds.shape[1]))), axis=0))
    gcn_net = GraphEncoder(graph=G,device=args.device, entity=env.user_length+env.item_length+env.feature_length+1+env.group_length, emb_size=embed.size(1), kg=kg, embeddings=embed, \
        fix_emb=args.fix_emb, seq=args.seq, gcn=args.gcn, hidden_size=args.hidden,u=env.user_length,v=env.item_length,f=env.feature_length,g=env.group_length).to(args.device)
    agent = Agent(device=args.device, memory=memory, state_size=args.hidden, action_size=embed.size(1), \
        hidden_size=args.hidden, gcn_net=gcn_net, learning_rate=args.learning_rate, l2_norm=args.l2_norm, PADDING_ID=embed.size(0)-1)
    if args.load_rl_epoch != 0 :
        print('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
        agent.load_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)

    test_performance = []
    if args.eval_num == 10:
        SR15_mean = dqn_evaluate(args, kg, agent, filename, 0)
        test_performance.append(SR15_mean)
    for train_step in range(1, args.max_steps+1):
        SR5, SR10, SR15, AvgT, Rank, NDCG, P, R, total_reward = 0., 0., 0., 0., 0., 0., 0., 0., 0.
        loss = torch.tensor(0, dtype=torch.float, device=args.device)
        for i_episode in tqdm(range(args.sample_times),desc='sampling'):
            blockPrint()
            print('\n================new tuple:{}===================='.format(i_episode))
            if not args.fix_emb:
                state, cand, action_space = env.reset(agent.gcn_net.embedding.weight.data.cpu().detach().numpy())  # Reset environment and record the starting state
            else:
                state, cand, action_space = env.reset() 
            #state = torch.unsqueeze(torch.FloatTensor(state), 0).to(args.device)
            
            epi_reward = 0
            is_last_turn = False
#             recom_items_ls = []
            for t in count():   # user  dialog
                if t == 14:
                    is_last_turn = True
                action, sorted_actions,state_emb = agent.select_action(state, cand, action_space, is_last_turn=is_last_turn)
                
                print('*****************************************')
                print(args.fix_emb)

                if not args.fix_emb:
                    next_state, next_cand, action_space, reward, done, recom_items, is_rec = env.step(action.item(), sorted_actions, agent.gcn_net.embedding.weight.data.cpu().detach().numpy())
                else:
                    next_state, next_cand, action_space, reward, done, recom_items, is_rec = env.step(action.item(), sorted_actions)
                epi_reward += reward
                reward = torch.tensor([reward], device=args.device, dtype=torch.float)
#                 if is_rec:
#                     enablePrint()
#                     recom_items_ls.append(recom_items)
                
                if done:
                    next_state = None
                # agent.memory.push(state, action, next_state, reward, next_cand,torch.cosine_similarity(state_emb[0],state_emb[1],dim=2))
                agent.memory.push(state, action, next_state, reward, next_cand)
                state = next_state
                cand = next_cand

                newloss = agent.optimize_model(args.batch_size, args.gamma)
                if newloss is not None:
                    loss += newloss

                if done:
                    # every episode update the target model to be same with model
                    if reward.item() == 1:  # recommend successfully
                        if t < 5:
                            SR5 += 1
                            SR10 += 1
                            SR15 += 1
                        elif t < 10:
                            SR10 += 1
                            SR15 += 1
                        else:
                            SR15 += 1
                        Rank += (1/math.log(t+3,2) + (1/math.log(t+2,2)-1/math.log(t+3,2))/math.log(done+1,2))
                        NDCG += cal_NDCG(t, recom_items, kg, env.user_id)
                        precision, recall = cal_P_R(recom_items, kg , env.user_id, env.ui_dict)
                        P += precision
                        R += recall
                    else:
                        Rank += 0
                        NDCG += 0
                        P += 0
                        R += 0
                    AvgT += t+1
                    total_reward += epi_reward
                    break
        
#         precision, recall = cal_P_R(recom_items_ls, kg , env.user_id)
        enablePrint() # Enable print function
        print('loss : {} in epoch_uesr {}'.format(loss.item()/args.sample_times, args.sample_times))
        print('SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{}, NDCG:{}, Precision:{}, Recall:{}, rewards:{} '
                  'Total epoch_uesr:{}'.format(SR5 / args.sample_times, SR10 / args.sample_times, SR15 / args.sample_times,
                                                AvgT / args.sample_times, Rank / args.sample_times, NDCG / args.sample_times, P / args.sample_times, R / args.sample_times, total_reward / args.sample_times, args.sample_times))

        if train_step % args.eval_num == 0:
            SR5_mean, SR10_mean, SR15_mean, AvgT_mean, Rank_mean, NDCG_mean, P_mean, R_mean = dqn_evaluate(args, kg, agent, filename, train_step)
            
            if SR15_mean>SR15_best:
                SR5_best, SR10_best, SR15_best, AvgT_best, Rank_best, NDCG_best, P_best, R_best=SR5_mean, SR10_mean, SR15_mean, AvgT_mean, Rank_mean, NDCG_mean, P_mean, R_mean
            print("best!!!!!!!!!SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{}, NDCG:{}, Precision:{}, Recall:{}!!!!".format(
                SR5_best, SR10_best, SR15_best, AvgT_best, Rank_best, NDCG_best, P_best, R_best))
            test_performance.append(SR15_mean)
        if train_step % args.save_num == 0:
            agent.save_model(data_name=args.data_name, filename=filename, epoch_user=train_step)
    print(test_performance)

class KG:
    def __init__(self):
        self.G = dict()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=1, help='random seed.')
    parser.add_argument('--gpu', type=str, default='1', help='gpu device.')
    parser.add_argument('--epochs', '-me', type=int, default=50000, help='the number of RL train epoch') # 
    parser.add_argument('--fm_epoch', type=int, default=0, help='the epoch of FM embedding')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--l2_norm', type=float, default=1e-6, help='l2 regularization.')
    parser.add_argument('--hidden', type=int, default=100, help='number of samples')
    parser.add_argument('--memory_size', type=int, default=5000, help='size of memory ')

    parser.add_argument('--data_name', type=str, default=ML1M, choices=[ LAST_FM_STAR, YELP_STAR,MOVIE,YELP_STAR,AMAZON],
                        help='One of { LAST_FM_STAR, YELP_STAR, BOOK,MOVIE}.')
    parser.add_argument('--entropy_method', type=str, default='match', help='entropy_method is one of {entropy, weight entropy,match}')
    # Although the performance of 'weighted entropy' is better, 'entropy' is an alternative method considering the time cost.
    parser.add_argument('--max_turn', type=int, default=15, help='max conversation turn')
    parser.add_argument('--attr_num', type=int, help='the number of attributes')
    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')
    parser.add_argument('--ask_num', type=int, default=1, help='the number of features asked in a turn')
    parser.add_argument('--load_rl_epoch', type=int, default=10, help='the epoch of loading RL model')

    parser.add_argument('--sample_times', type=int, default=100, help='the epoch of sampling') #100
    parser.add_argument('--max_steps', type=int, default=10, help='max training steps') #100
    
    parser.add_argument('--observe_num', type=int, default=100, help='the observe_num') #100
    parser.add_argument('--eval_num', type=int, default=1, help='the number of steps to evaluate RL model and metric')
    parser.add_argument('--choice_num', type=int, default=4, help='the choice_num')
    parser.add_argument('--save_num', type=int, default=10, help='the number of steps to save RL model and metric')
    parser.add_argument('--cand_num', type=int, default=10, help='candidate sampling number')
    parser.add_argument('--cand_item_num', type=int, default=10, help='candidate item sampling number')

    parser.add_argument('--fea_score', type=str, default='other', help='feature score')
    parser.add_argument('--fix_emb', action='store_false', help='fix embedding or not')
    parser.add_argument('--embed', type=str, default=None, help='pretrained embeddings,transe')
    parser.add_argument('--seq', type=str, default='transformer', choices=['rnn', 'transformer', 'mean'], help='sequential learning method')
    parser.add_argument('--gcn', action='store_false', help='use GCN or not')
    


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(args.device)
    print('data_set:{}'.format(args.data_name))
    
    kg = pickle.load(open('tmp/ml1m/kg.pkl', 'rb'))
#     kg = load_kg(args.data_name)
    #reset attr_num
    feature_name = FeatureDict[args.data_name]
    feature_length = len(kg.G[feature_name].keys())
    print('dataset:{}, feature_length:{}'.format(args.data_name, feature_length))
    args.attr_num = feature_length  # set attr_num  = feature_length
    print('args.attr_num:', args.attr_num)
    print('args.entropy_method:', args.entropy_method)

#     dataset = load_dataset(args.data_name)
    filename = 'train-data-{}-RL-cand_num-{}-cand_item_num-{}-embed-{}-seq-{}-gcn-{}'.format(
        args.data_name, args.cand_num, args.cand_item_num, args.embed, args.seq, args.gcn)
    train(args, kg, filename)
    
def cal_NDCG(t, recom_items, kg, n):
    K = len(recom_items)
    DCG = 0
    IDCG = 0
    
    users = kg.G['group'][n]['member_of']
    rel = []
    for i in recom_items:
        num_m = 0
        for user in users:
            if i in kg.G['user'][user]['interact']:
                num_m += 1
        rel.append(num_m)
    for k in range(1,K+1):
        DCG += rel[k-1] / math.log(k+1, 2)
    ideal_rel = sorted(rel, reverse = True)
    for k in range(1,K+1):
        IDCG += ideal_rel[k-1] / math.log(k+1, 2)
    if IDCG != 0:
        return DCG / IDCG
    else:
        return 0

def cal_P_R(recom_items, kg, n, gi):
    items = list(set(gi[str(n)]))
    hit = 0
    for i in recom_items:
        if i in items:
            hit += 1
    return hit / len(recom_items), hit / len(items)
    
# def cal_P_R(recom_items_ls, kg, group_id):
#     items = kg.G['group'][group_id]['group_interact']
#     precision_ls = []
#     recall_ls = []
#     for rec_i in recom_items_ls:
#         hit = 0
#         for i in rec_i:
#             if i in items:
#                 hit += 1
#         precision_ls.append(hit / len(rec_i))
#         recall_ls.append(hit / len(items))
#     return np.mean(precision_ls), np.mean(recall_ls)

if __name__ == '__main__':
    main()

