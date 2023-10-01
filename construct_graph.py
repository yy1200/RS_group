from utils import *
from tqdm import tqdm

def get_graph(a,b,c,d,data):

    def load_kg(dataset=TMP_DIR[data]):
        kg_file = dataset + '/kg.pkl'
        kg = pickle.load(open(kg_file, 'rb'))
        return kg
    kg=load_kg()
    
    u_gu=[]
    v_gu=[]
    for n in tqdm(kg.G['group']):
        for item in kg.G['group'][n]['member_of']:
            u_gu.append(int(n))
            v_gu.append(int(item))
    u_gi=[]
    v_gi=[]
    for n in tqdm(kg.G['group']):
        for item in kg.G['group'][n]['group_interact']:
            u_gi.append(int(n))
            v_gi.append(int(item))
    u_ga=[]
    v_ga=[]
    for n in tqdm(kg.G['group']):
        for item in kg.G['group'][n]['group_like']:
            u_ga.append(int(n))
            v_ga.append(int(item))
    
    u_ui=[]
    v_ui=[]
    for n in tqdm(kg.G['user']):
        for item in kg.G['user'][n]['interact']:
            u_ui.append(int(n))
            v_ui.append(int(item))
    u_ua=[]
    v_ua=[]
    for n in tqdm(kg.G['user']):
        for item in kg.G['user'][n]['like']:
            u_ua.append(int(n))
            v_ua.append(int(item))
    u_ia=[]
    v_ia=[]
    for n in tqdm(kg.G['item']):
        for item in kg.G['item'][n]['belong_to']:
            u_ia.append(int(n))
            v_ia.append(int(item))

    G = dgl.heterograph({
        ('group', 'member_of', 'user'): (u_gu, v_gu),
        ('user', 'member_of', 'group'): (v_gu, u_gu),
        ('group', 'group_interact', 'item'): (u_gi, v_gi),
        ('item', 'group_interact', 'group'): (v_gi, u_gi),
        ('group', 'group_like', 'attribute'): (u_ga, v_ga),
        ('attribute', 'group_like', 'group'): (v_ga, u_ga),
        ('user', 'interact', 'item'): (u_ui, v_ui),
        ('item', 'interact', 'user'): (v_ui,u_ui),
        ('user', 'like', 'attribute'): (u_ua, v_ua),
        ('attribute', 'like', 'user'): (v_ua,u_ua),
        ('item', 'belong_to', 'attribute'): (u_ia, v_ia),
        ('attribute', 'belong_to', 'item'): (v_ia,u_ia),
        
    },num_nodes_dict={'item':b,'user':a,'attribute':c+1,'group':d})
    print("item:{},user:{},feature:{},group:{}".format(b,a,c,d))
    return G