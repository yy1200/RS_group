import time
import argparse
from itertools import count
import torch.nn as nn
import torch
import math
from collections import namedtuple
from utils import *
from RL.env_multi_choice_question import MultiChoiceRecommendEnv
from tqdm import tqdm
EnvDict = {
        LAST_FM_STAR: MultiChoiceRecommendEnv,
        YELP_STAR: MultiChoiceRecommendEnv,
        BOOK:MultiChoiceRecommendEnv,
        MOVIE:MultiChoiceRecommendEnv,
        AMAZON:MultiChoiceRecommendEnv,
        ML1M:MultiChoiceRecommendEnv,
    }

def dqn_evaluate(args, kg, agent, filename, i_episode):
    test_env = EnvDict[args.data_name](kg, args.data_name, args.embed, seed=args.seed, max_turn=args.max_turn,
                                       cand_num=args.cand_num, cand_item_num=args.cand_item_num, attr_num=args.attr_num, mode='test', ask_num=args.ask_num, entropy_way=args.entropy_method,
                                       fm_epoch=args.fm_epoch)
    set_random_seed(args.seed)
    tt = time.time()
    start = tt
    SR5, SR10, SR15, AvgT, Rank, total_reward, NDCG, P, R = 0, 0, 0, 0, 0, 0, 0, 0, 0
    SR_turn_15 = [0]* args.max_turn
    turn_result = []
    result = []
    user_size = test_env.ui_array.shape[0]
    print('User size in UI_test: ', user_size)
    test_filename = 'Evaluate-epoch-{}-'.format(i_episode) + filename
    plot_filename = 'Evaluate-'.format(i_episode) + filename
    if args.data_name in [LAST_FM_STAR,]:
        if args.eval_num == 1:
            test_size = 500
        else:
            test_size = 4000     # Only do 4000 iteration for the sake of time
        user_size = test_size
    else:
        if args.eval_num == 1:
            test_size = 500 # 500
        else:
            test_size = 100 # 2500     # Only do 2500 iteration for the sake of time
        user_size = test_size
    print('The select Test size : ', test_size)
    for user_num in tqdm(range(user_size)):  #user_size
        # TODO uncommend this line to print the dialog process
        blockPrint()
        print('\n================test tuple:{}===================='.format(user_num))
        if not args.fix_emb:
            state, cand, action_space = test_env.reset(agent.gcn_net.embedding.weight.data.cpu().detach().numpy())  # Reset environment and record the starting state
        
        else:
            state, cand, action_space = test_env.reset() 
        epi_reward = 0
        is_last_turn = False
        
#         recom_items_ls = []
        for t in count():  # user  dialog
            if t == 14:
                is_last_turn = True
            action, sorted_actions,_ = agent.select_action(state, cand, action_space, is_test=True, is_last_turn=is_last_turn)
            next_state, next_cand, action_space, reward, done, recom_items, is_rec = test_env.step(action.item(), sorted_actions)
            
#             if is_rec:
#                 recom_items_ls.append(recom_items)
            epi_reward += reward
            reward = torch.tensor([reward], device=args.device, dtype=torch.float)
            if done:
                next_state = None
            state = next_state
            cand = next_cand
            if done:
#                 enablePrint()
                if reward.item() == 1:  # recommend successfully
                    SR_turn_15 = [v+1 if i>t  else v for i, v in enumerate(SR_turn_15) ]
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
                    NDCG += cal_NDCG(t, recom_items, kg, test_env.user_id)
                    precision, recall = cal_P_R(recom_items, kg , test_env.user_id, test_env.ui_dict)
                    P += precision
                    R += recall
                else:
                    Rank += 0
                    NDCG += 0
                    P += 0
                    R += 0
                total_reward += epi_reward
                AvgT += t+1
                break
                
#         precision, recall = cal_P_R(recom_items_ls, kg , test_env.user_id, test_env.ui_dict)
        
        if (user_num+1) % args.observe_num == 0 and user_num > 0:
            SR = [SR5/args.observe_num, SR10/args.observe_num, SR15/args.observe_num, AvgT / args.observe_num, Rank / args.observe_num, NDCG / args.observe_num, P / args.observe_num, R / args.observe_num, total_reward / args.observe_num]
            SR_TURN = [i/args.observe_num for i in SR_turn_15]
            print('Total evalueation epoch_uesr:{}'.format(user_num + 1))
            print('Takes {} seconds to finish {}% of this task'.format(str(time.time() - start),
                                                                       float(user_num) * 100 / user_size))
            print('SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{}, NDCG:{}, Precision:{}, Recall:{}, reward:{} '
                  'Total epoch_uesr:{}'.format(SR5 / args.observe_num, SR10 / args.observe_num, SR15 / args.observe_num,
                                                AvgT / args.observe_num, Rank / args.observe_num, NDCG / args.observe_num, P / args.observe_num, R / args.observe_num, total_reward / args.observe_num, user_num + 1))
            
            result.append(SR)
            turn_result.append(SR_TURN)
            SR5, SR10, SR15, AvgT, Rank, NDCG, P, R, total_reward = 0, 0, 0, 0, 0, 0, 0, 0, 0
            SR_turn_15 = [0] * args.max_turn
            tt = time.time()
#         enablePrint()   
    
    SR5_mean = np.mean(np.array([item[0] for item in result]))
    SR10_mean = np.mean(np.array([item[1] for item in result]))
    SR15_mean = np.mean(np.array([item[2] for item in result]))
    AvgT_mean = np.mean(np.array([item[3] for item in result]))
    Rank_mean = np.mean(np.array([item[4] for item in result]))
    NDCG_mean = np.mean(np.array([item[5] for item in result]))
    P_mean = np.mean(np.array([item[6] for item in result]))
    R_mean = np.mean(np.array([item[7] for item in result]))
    reward_mean = np.mean(np.array([item[8] for item in result]))
    SR_all = [SR5_mean, SR10_mean, SR15_mean, AvgT_mean, Rank_mean, NDCG_mean, P_mean, R_mean, reward_mean]
    save_rl_mtric(dataset=args.data_name, filename=filename, epoch=user_num, SR=SR_all, spend_time=time.time() - start,
                  mode='test')
    save_rl_mtric(dataset=args.data_name, filename=test_filename, epoch=user_num, SR=SR_all, spend_time=time.time() - start,
                  mode='test')  # save RL SR
    print('save test evaluate successfully!')

    print('SR5:{}, SR10:{}, SR15:{}, AvgT:{}, Rank:{}, NDCG:{}, Precision:{}, Recall:{}, reward:{}'.format(SR5_mean, SR10_mean, SR15_mean, AvgT_mean, Rank_mean, NDCG_mean, P_mean, R_mean, reward_mean))
    PATH = TMP_DIR[args.data_name] + '/RL-log-merge/' + test_filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('Training epocch:{}\n'.format(i_episode))
        f.write('===========Test Turn===============\n')
        f.write('Testing {} user tuples\n'.format(user_num))
        f.write('================================\n')
    PATH = TMP_DIR[args.data_name] + '/RL-log-merge/' + plot_filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(i_episode, SR15_mean, AvgT_mean, Rank_mean, NDCG_mean, P_mean, R_mean, reward_mean))

    return SR5_mean, SR10_mean, SR15_mean, AvgT_mean, Rank_mean, NDCG_mean, P_mean, R_mean

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

    