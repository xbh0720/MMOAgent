from .LLM_chat import llm_server
import re
import random
import logging
import numpy as np 
import json
from collections import deque
from copy import deepcopy


class LLM_Agent3:
    def __init__(self, agent_id, player_profile = None) -> None:
        self.action_space = ['Upgrade', 'Task', 'Shop', 'Recharge', 'Sell', 'Buy'] #['Upgrade', 'Task', 'Shop', 'Recharge', 'Market'] 
        self.one_step = ['Upgrade', 'Recharge']
        self.visible_width = 5
        self.reverse_dict = {'Exp':'Experience', 'Mat':'Material'}
        self.costs = {'Upgrade':{'Experience':1, 'Material':1, 'Token':1}, 'Shop':{'Token':10}, 'Recharge':{'Currency':1}, 'Buy':{'Token':1}, 'Sell':{'Material':1}} #'Market':{'Buy':{'Token':1}, 'Sell':{'Material':1}}
        self.price_level = 10
        self.profile = player_profile
        self.agent_id = agent_id
        self.bargain_state_all = {}  
        self.seller_init_message = None
        self.min_ask_price = 100
        self.online_step = 0
        self.total_online_steps = 0
        self.clock_step = 0
        self.reflect_cycle = 10
        self.initial_bargain_content = "Hi, the material's price is {} Tokens."
        self.last_direction = None
        self.tax_rate = 0.1
        self.visited_positions = set()
        self.task_continue = 0

    def reset_memory(self):
        self.trajectory_memory = []

    
    def get_center(self, resource, map, agent_loc):
        r, c = agent_loc
        rows, columns = map.shape
        self.map_height, self.map_width = rows, columns
        w = self.visible_width
        left_margin = np.maximum(0, c - w)
        right_margin = np.minimum(columns - 1, c + w)
        up_margin = np.maximum(0, r - w)
        down_margin = np.minimum(rows - 1, r + w)
        descriptions = []
        locs = []
        relative_locs = []
        values = []
        for i in range(up_margin, down_margin + 1):
            for j in range(left_margin, right_margin + 1):
                 if map[i, j] > 0:
                     # resource, row, column, value
                    loc = (i, j)
                    relative_loc = (i - r, j - c)
                    descriptions.append(f'There are {map[i][j]} {resource} at location {relative_loc}.')
                    locs.append(loc)
                    relative_locs.append(relative_loc)
                    values.append(map[i,j])
        
        return descriptions, relative_locs, values

    def generate_obs(self, resources_maps, agent_loc):
        resource = resources_maps.keys()
        resource_obs = {}
        for resource, map in resources_maps.items():
            rev_name = self.reverse_dict[resource] if resource in self.reverse_dict.keys() else resource
            description, locs, values = self.get_center(rev_name, map, agent_loc)
            if len(values) > 0:    
                resource_obs[rev_name] = {}
                resource_obs[rev_name]['description'] = description
                resource_obs[rev_name]['locs'] = locs
                resource_obs[rev_name]['values'] = values
        
        return resource_obs

    def generate_direction(self, nearest_res_loc):
        if not nearest_res_loc:
            relative_locs = [(0,-1),(0,1),(-1,0),(1,0)]
            relative_direction = ['Left', 'Right', 'Up', 'Down']
            self.visited_positions.add(tuple(self.agent_loc))
            task_mask = self.masks['Task']
            indices = [0,1,2,3]
            random.shuffle(indices)
            for i in indices:
                if task_mask[i]:
                    dx, dy = relative_locs[i]
                    new_loc = (self.agent_loc[0] + dx, self.agent_loc[1] + dy)
                    if not new_loc in self.visited_positions:
                        return relative_direction[i]
            return random.choice(relative_direction)
        else:
            self.visited_positions = set()
            self.last_direction = None
            relative_r, relative_c = nearest_res_loc
        if relative_r < 0:
            return 'Up'
        elif relative_r > 0:
            return 'Down'
        else:
            if relative_c < 0:
                return 'Left'
            else:
                return 'Right'

    def generate_shop_decision(self, inventory):
        if inventory['Experience'] < inventory['Material']:
            return 'Experience'
        else:
            return 'Material'
        
    def enough_inventory(self, inventory, action):
        if action in self.costs.keys():
            action_cost = self.costs[action]
            for resource, cost in action_cost.items():
                if cost > inventory[resource]:
                    return False
        return True
    
    def get_min_price(self, price_hist):
        for i in range(np.shape(price_hist)[0]):
            if price_hist[i] > 0:
                return i
        return None
    
    def get_max_price(self, price_hist):
        for i in reversed(range(np.shape(price_hist)[0])):
            if price_hist[i] > 0:
                return i
        return None
    
    def generate_first_legal_action(self, req):
        tolerance = 10
        is_enough = None
        while not is_enough and tolerance > 0:   
            action_content = random.choice(self.action_space)
            tolerance -= 1
            #print(response)
            is_enough = self.enough_inventory(req['inventory'], action_content)

        return action_content


    def generate_action(self, inventory, escrow, capability, rseource_maps, agent_locs, market_hist, bargain_message = None, nearest = True, masks= None):
        action_text = None
        self.agent_loc = agent_locs[self.agent_id]
        self.masks = masks
        # handle messages
        req = {}
        resource_obs = self.generate_obs(rseource_maps, agent_locs[self.agent_id])
        resource_obs_summary = {}
        resource_locs = []
        resource_desps = []
        self.clock_step += 1
        for resource in resource_obs.keys():
            res_sum_value = sum(resource_obs[resource]['values'])
            resource_obs_summary[resource] = res_sum_value
            resource_locs.extend(resource_obs[resource]['locs'])
            resource_desps.extend(resource_obs[resource]['description'])

        req['inventory'] = { (self.reverse_dict[k] if k in self.reverse_dict.keys() else k):v for k,v in inventory.items()}
        req['escrow'] = { (self.reverse_dict[k] if k in self.reverse_dict.keys() else k):v for k,v in escrow.items() if k in ['Mat', 'Token']}
        req['actions'] = self.action_space
        req['capability'] = capability


        if action_text == None:
            if self.task_continue > 0:
                action_content = 'Task'
                self.task_continue -= 1
            else:
                action_content = self.generate_first_legal_action(req)
                if action_content == 'Task':
                    self.task_continue = 5

            if action_content in self.one_step:
                action_text =  action_content
            elif action_content == 'Task':

                sorted_indexes = sorted(range(len(resource_locs)), key=lambda i: abs(resource_locs[i][0]) + abs(resource_locs[i][1]))[:3]
                task_decision = self.generate_direction(resource_locs[sorted_indexes[0]] if len(sorted_indexes) > 0 else None)

                action_text =  action_content + '+' + task_decision
            elif action_content == 'Shop':
                
                shop_decision = self.generate_shop_decision(req['inventory'])
                action_text = action_content + '+' + shop_decision
            elif action_content == 'Sell':     
                price = random.randint(1, 10)
                action_text = 'Market' + '.' + action_content + '_' + 'Mat' + '+' + str(price) 

            elif action_content == 'Buy':
                border = min(req['inventory']['Token'], 10)
                if border > 1: 
                    price = random.randint(1, border)
                else:
                    price = 1
                action_text = 'Market' + '.' + action_content + '_' + 'Mat' + '+' + str(price)
                
            else:
                action_text = 0
        
        action = self.parse_action(action_text)
        return action
    
    
    def parse_action(self, action_text):
        if action_text == 0:
            return 0
        if '+' in action_text:
            component_name = action_text.split('+')[0]
            action_ = action_text.split('+')[1]
        else:
            component_name = action_text
        if component_name in self.one_step:
            action_num = 1
        elif component_name == 'Task':
            reverse = {'Left':1, 'Right':2, 'Up':3, 'Down':4}
            action_num = reverse[action_]
        elif component_name == 'Shop':
            reverse2 = {'Experience':1, 'Material':2}
            action_num = reverse2[action_]
        elif component_name == 'Bargain':
            if action_.isdigit():
                action_num = int(action_) + 1  #deal price
            else:
                action_num = json.loads(action_) #response message
        else:
            action_num = int(action_) + 1
        
        return {component_name : action_num}