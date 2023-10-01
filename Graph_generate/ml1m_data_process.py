import os
import json
from easydict import EasyDict as edict

class Ml1mDataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir + '/Graph_generate_data'
        self.load_entities()
        self.load_relations()
    def get_relation(self):
        #Entities
        GROUP = 'group'
        USER = 'user'
        ITEM = 'item'
        FEATURE = 'feature'
        LARGE_FEATURE = 'large_feature'

        #Relations
        MEMBER_OF = 'member_of'
        GROUP_INTERACT= 'group_interact'
        GROUP_LIKE = 'group_like'
        INTERACT = 'interact'
        FRIEND = 'friends'
        LIKE = 'like'
        BELONG_TO = 'belong_to'
        BELONG_TO_LARGE = 'belong_to_large'  # feature(second-layer tag) --> large_feature(first-layer tag)
        LINK_TO_FEATURE = 'link_to_feature'   # large_feature(first-layer tag)  -->  feature(second-layer tag)
        relation_name = [MEMBER_OF, GROUP_INTERACT, GROUP_LIKE, INTERACT, FRIEND, LIKE, BELONG_TO, BELONG_TO_LARGE, LINK_TO_FEATURE]

        fm_relation = {
            GROUP: {
                MEMBER_OF: USER,
                GROUP_INTERACT: ITEM,
                GROUP_LIKE: FEATURE
            },
            USER: {
                INTERACT: ITEM,
                FRIEND: USER, #There is no such relationship in ml1m
                LIKE: FEATURE  
            },
            ITEM: {
                BELONG_TO: FEATURE,
                BELONG_TO_LARGE: LARGE_FEATURE,
                INTERACT: USER
            },
            FEATURE: {
                LIKE: USER,
                BELONG_TO: ITEM,
                LINK_TO_FEATURE: LARGE_FEATURE
            },
            LARGE_FEATURE: {
                LIKE: USER,
                BELONG_TO_LARGE: ITEM,
                LINK_TO_FEATURE: FEATURE

            }

        }
        relation_link_entity_type = {
            MEMBER_OF: [GROUP, USER],
            GROUP_INTERACT: [GROUP, ITEM],
            GROUP_LIKE: [GROUP, FEATURE],
            INTERACT:  [USER, ITEM],
            FRIEND:  [USER, USER],
            LIKE:  [USER, FEATURE],
            BELONG_TO:  [ITEM, FEATURE],
            BELONG_TO_LARGE:  [ITEM, LARGE_FEATURE],
            LINK_TO_FEATURE: [LARGE_FEATURE, FEATURE]
        }
        return fm_relation, relation_name, relation_link_entity_type
    def load_entities(self):
        entity_files = edict(
            group='group_user.json',
            user='user_dict.json',
            item='item_dict-original_tag.json',
            feature='second-layer_oringinal_tag_map.json',
            large_feature='first-layer_merged_tag_map.json'
        )
        for entity_name in entity_files:
            with open(os.path.join(self.data_dir,entity_files[entity_name]), encoding='utf-8') as f:
                mydict = json.load(f)
            if entity_name in ['feature']:
                entity_id = list(mydict.values())
            elif entity_name in ['large_feature']:
                entity_id = list(map(int, list(mydict.values())))
            else:
                entity_id = list(map(int, list(mydict.keys())))
            setattr(self, entity_name, edict(id=entity_id, value_len=max(entity_id)+1))
            print('Load', entity_name, 'of size', len(entity_id))
            print(entity_name, 'of max id is', max(entity_id))

    def load_relations(self):
        """
        relation: head entity---> tail entity
        --
        """
        Ml1m_relations = edict(
            member_of=('group_user.json', self.group, self.user),
            group_interact=('group_item.json', self.group, self.item),
            group_like=('group_attribute.json', self.group, self.feature),
            interact=('user_item.json', self.user, self.item), #(filename, head_entity, tail_entity)
            friends=('user_dict.json', self.user, self.user),
            like=('user_dict.json', self.user, self.feature),
            belong_to=('item_dict-original_tag.json', self.item, self.feature),
            belong_to_large=('item_dict-merged_tag.json', self.item, self.large_feature),
            link_to_feature=('2-layer taxonomy.json', self.large_feature, self.feature)
        )
        for name in Ml1m_relations:
            #  Save tail_entity
            relation = edict(
                data=[],
            )
            knowledge = [list([]) for i in range(Ml1m_relations[name][1].value_len)]
            # load relation files
            with open(os.path.join(self.data_dir, Ml1m_relations[name][0]), encoding='utf-8') as f:
                mydict = json.load(f)
            if name in ['member_of', 'group_interact', 'group_like', 'interact', 'belong_to_large']:
                for key, value in mydict.items():
                    head_id = int(key)
                    tail_ids = value
                    knowledge[head_id] = tail_ids
            elif name in ['friends', 'like']:
                for key in mydict.keys():
                    head_str = key
                    head_id = int(key)
                    tail_ids = mydict[head_str][name]
                    knowledge[head_id] = tail_ids
            elif name in ['belong_to']:
                for key in mydict.keys():
                    head_str = key
                    head_id = int(key)
                    tail_ids = mydict[head_str]['feature_index']
                    knowledge[head_id] = tail_ids
            elif name in ['link_to_feature']:
                with open(os.path.join(self.data_dir, 'first-layer_merged_tag_map.json'), encoding='utf-8') as f:
                    tag_map = json.load(f)
                for key, value in mydict.items():
                    head_id = tag_map[key]
                    tail_ids = value
                    knowledge[head_id] = tail_ids
            relation.data = knowledge
            setattr(self, name, relation)
            tuple_num = 0
            for i in knowledge:
                tuple_num += len(i)
            print('Load', name, 'of size', tuple_num)

