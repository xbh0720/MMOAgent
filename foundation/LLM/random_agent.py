from .LLM_chat import llm_server
import re
import random
import logging
import numpy as np 
import json
from collections import deque
from copy import deepcopy


class LLM_Agent4:
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


    def generate_direction(self):
        direction = random.choice(["Left", "Right", "Up", "Down"])
        return direction

    def generate_shop_decision(self):
        return random.choice(["Experience", "Material"])
        
    def enough_inventory(self, inventory, action):
        if action in self.costs.keys():
            action_cost = self.costs[action]
            for resource, cost in action_cost.items():
                if cost > inventory[resource]:
                    return False
        return True
    
    
    def generate_first_legal_action(self, req):
        action_content = random.choice(self.action_space)

        return action_content


    def generate_action(self, inventory, escrow, capability, rseource_maps, agent_locs, market_hist, bargain_message = None, nearest = True, masks = None):
        action_text = None
        req = {}
        req['inventory'] = { (self.reverse_dict[k] if k in self.reverse_dict.keys() else k):v for k,v in inventory.items()}
        
        if action_text == None:
            action_content = self.generate_first_legal_action(req)

            if action_content in self.one_step:
                action_text =  action_content
            elif action_content == 'Task': 
                task_decision = self.generate_direction()
                action_text =  action_content + '+' + task_decision
            elif action_content == 'Shop':
                shop_decision = self.generate_shop_decision()
                action_text = action_content + '+' + shop_decision
            
            elif action_content == 'Sell':    
                price = random.randint(1, 10)
                action_text = 'Market' + '.' + action_content + '_' + 'Mat' + '+' + str(price) 

            elif action_content == 'Buy':
                border = min(inventory["Token"], 10)
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
                action_num = int(action_) + 1 
            else:

                action_num = json.loads(action_) 
        else:
            action_num = int(action_) + 1
        
        return {component_name : action_num}