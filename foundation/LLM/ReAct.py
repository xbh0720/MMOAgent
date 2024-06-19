from .LLM_chat import llm_server
import re
import random
import logging
import numpy as np 
import json
from collections import deque
from copy import deepcopy
from datetime import datetime
# LLM choose proper action through observation, action格式需得和base-agent中parse_action一致，可以是{subspace：action}的形式
class ReAct:
    def __init__(self, agent_id, player_profile = None) -> None:
        self.background = '''You plays a role as a game player. {profile}. The game encompasses three primary resources: Experience, Token, and Material, which are both obtainable and expendable in the game. You start with a fixed amount of Currency (i.e. money), used for recharging and not acquirable afterward.
There are 7 kinds of activities you could act in the game as follows:
1. Upgrade: Expending 1 Experience, 1 Token, and 1 Material to improve role's capability.
2. Task: Collecting nearby Experience, Token, or Material with much effort by navigating the game map.
3. Shop: Spending 10 Tokens to purchase 1 Experience or 1 Material from the in-game mall.
4. Recharge: Using 1 Currency to acquire 10 Tokens when lacking token.
5. Sell: Selling Material to other players for Tokens.
6. Buy: Buying Material from auctions using Tokens, potentially cheaper than the Shop but requires waiting for sellers and bidding with other players.
Here is an example of chosen one specific activity as output based on the input past actions and observations:
Input:
The following is the game state you observe. Your inventory: {{'Currency': 10, 'Experience': 0, 'Material': 0, 'Token': 0}}. Your escrowment in auction: {{'Material': 0, 'Token': 0}}, with materials on selling and tokens on bidding and the escrowment resources cannot be used elsewhere. Your capability: 0. Auction info: 0 materials on selling with the lowest selling price at None tokens. 0 puchasing needs from buyers, with the highest bidding price at None tokens. The past average deal price in auction is None tokens. Nearby resources: {{'Experience': 2.0, 'Token': 3.0}}.
> Think: I have no game resources for upgrade in initial, but I have enough currency and there are rich resources nearby, so I could recharge or do task first to obtain resources. 
> Action: Recharge
Your inventory: {{'Currency': 9, 'Experience': 0, 'Material': 0, 'Token': 10}}. Your escrowment in auction: {{'Material': 0, 'Token': 0}}, with materials on selling and tokens on bidding and the escrowment resources cannot be used elsewhere. Your capability: 0. Auction info: 0 materials on selling with the lowest selling price at None tokens. 0 puchasing needs from buyers, with the highest bidding price at None tokens. The past average deal price in auction is None tokens. Nearby resources: {{'Experience': 2.0, 'Token': 3.0}}.
> Think: My game resources is still not enough for upgrade, but I can do task to collect resources. 
> Action: Task
Your inventory: {{'Currency': 9, 'Experience': 1, 'Material': 0, 'Token': 10}}. Your escrowment in auction: {{'Material': 0, 'Token': 0}}, with materials on selling and tokens on bidding and the escrowment resources cannot be used elsewhere. Your capability: 0. Auction info: 1 materials on selling with the lowest selling price at 8 tokens. 0 puchasing needs from buyers, with the highest bidding price at None tokens. The past average deal price in auction is None tokens. Nearby resources: {{'Experience': 1.0, 'Token': 3.0}}. 
> Think: I lack material for upgrading. There are no materials nearby for collecting but one material is selling 8 tokens which is not so expensive and I have enough tokens to buy it in the auction. 
> Action: Buy with 8 tokens
Your inventory: {{'Currency': 9, 'Experience': 1, 'Material': 1, 'Token': 2}}. Your escrowment in auction: {{'Material': 0, 'Token': 0}}, with materials on selling and tokens on bidding and the escrowment resources cannot be used elsewhere. Your capability: 0. Auction info: 0 materials on selling with the lowest selling price at None tokens. 0 puchasing needs from buyers, with the highest bidding price at None tokens. The past average deal price in auction is None tokens. Nearby resources: {{'Experience': 1.0, 'Token': 3.0}}. 
Output:
Think: I have enough resources for upgrade which cost 1 Experience, 1 Token, and 1 Material.
Action: Upgrade
=========================
You should first think about what your next action should be and then respond with one chosen activity among {actions}. Please note if your choose Buy or Sell, you should use " with xx tokens" to represent your price.  If no resources are nearby, you can also further explore through Task.
Input:
{history_str}
Output:
'''
        self.obs_template = """
            Your inventory: {inventory}. Your escrowment in auction: {escrow}, with materials on selling and tokens on bidding and the escrowment resources cannot be used elsewhere. Your capability: {capability}. Auction info: {sell_num} materials on selling with the lowest selling price at {lowest_price} tokens. {bid_num} puchasing needs from buyers, with the highest bidding price at {highest_price} tokens. The past average deal price in auction is {avg_market_rate} tokens. Nearby resources: {resource_obs_summary}.
            """
        
        self.memory = deque(maxlen=6) 
        self.Prompt_Config = {
            "stop": None,
            "temperature": 0.3,
            "maxTokens": 500, 
            "model": 'gpt-3.5-turbo'# "gpt-3.5-turbo-instruct",
        }
        self.action_space = ['Upgrade', 'Task', 'Shop', 'Recharge', 'Sell', 'Buy'] 
        self.one_step = ['Upgrade', 'Recharge']
        self.visible_width = 5 
        self.reverse_dict = {'Exp':'Experience', 'Mat':'Material'}
        
        self.costs = {'Upgrade':{'Experience':1, 'Material':1, 'Token':1}, 'Shop':{'Token':10}, 'Recharge':{'Currency':1}, 'Buy':{'Token':1}, 'Market.Sell_Mat':{'Material':1}} #'Market':{'Buy':{'Token':1}, 'Sell':{'Material':1}}
        self.price_level = 10
        self.profile = player_profile
        self.agent_id = agent_id
        
        self.bargain_state_all = {}  
        self.seller_init_message = None
        self.min_ask_price = 100
        self.online_step = 0
        self.total_online_steps = 0
        self.reflect_cycle = 10
        self.initial_bargain_content = "Hi, the material's price is {} Tokens."
        self.last_direction = None
        self.tax_rate = 0.1
        self.action_history = deque(maxlen=5)
        self.obs_history = deque(maxlen=5)
        self.reason_history = deque(maxlen=5)
        self.last_illegal_flag = False
        self.LLM_number = 0
        self.task_continue = 0
        self.task_finish = True

   
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
            self.task_continue += 1
            if self.task_continue >= 5:
                self.task_finish = True
            relative_locs = [(0,-1),(0,1),(-1,0),(1,0)]
            relative_direction = ['Left', 'Right', 'Up', 'Down']
            self.visited_positions.add(tuple(self.agent_loc))
            indices = [0,1,2,3]
            random.shuffle(indices)
            for i in indices:
                dx, dy = relative_locs[i]
                new_loc = (self.agent_loc[0] + dx, self.agent_loc[1] + dy)
                if not new_loc in self.visited_positions:
                    return relative_direction[i]
            return random.choice(relative_direction)
        else:
            self.visited_positions = set()
            # self.last_direction = None
            relative_r, relative_c = nearest_res_loc
            if abs(relative_r) + abs(relative_c) <= 1:
                self.task_finish = True
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
    
    def get_LLM_response(self, prompt_str, prompt_config = None):
        if not prompt_config:
            prompt_config = self.Prompt_Config
        retry = 3
        while retry > 0:
            retry -= 1
            try:
                response, _, actual_prompt = llm_server(
                {"prompt": prompt_str, **prompt_config}, mode = 'chat'
                )
                logging.info(str(actual_prompt) + '\\' + '\n' + response + '\\')
                self.LLM_number += 1
                return response, actual_prompt
            except:
                print('LLM call failed!')
        return "", ""
        

    def form_obs_history(self, obs_history, action_history, reason_history, current_obs):
        obs_str = "The following is the game state you observe."
        for i in range(len(obs_history)):
            obs_str = obs_str + obs_history[i] + '\n' + '> Think: ' + reason_history[i] + '\n' + '> Action: ' + action_history[i] + '\n'
            # obs_str += "Your game state changed."
        obs_str += current_obs
        return obs_str

    def test_enough(self, action_content, inventory):
        if action_content in self.costs.keys():
            action_cost = self.costs[action_content]
            for resource, cost in action_cost.items():
                if cost > inventory[resource]:
                    return False
        elif "buy" in action_content.lower():
            price = int(action_content.split('+')[1])
            if price > inventory["Token"]:
                return False
        elif "sell" in action_content.lower():
            if inventory["Material"] <= 0:
                return False
        return True

    def parse_response(self, response):
        action_match = re.search(r'Action:(.*)', response)
        reason_match = re.search(r'Think:(.*)', response)
        if action_match and reason_match:
            llm_action_response = action_match.group()
            reason = reason_match.group(1)
        else:
            llm_action_response = "Task"
            reason = "I need to collect more resources for upgrade."
        find_action = False
        for legal_action in self.action_space:
            if legal_action.lower() in llm_action_response.lower():
                find_action = True
                if legal_action in ["Buy", "Sell"]:
                    price_match = re.search(r'\d+\.*\d*', llm_action_response)
                    if price_match:
                        try:
                            price = int(float(price_match.group()))
                            if price > 0 and price < self.price_level:
                                pass
                            else:
                                price = random.randint(1, 10)
                        except:
                                price = random.randint(1, 10)
                    else:
                        price = random.randint(1, 10)
                    action_content = 'Market' + '.' + legal_action + '_' + 'Mat' + '+' + str(price)
                    action_response = legal_action + ' with {} tokens'.format(price)
                else:
                    action_content = legal_action
                    action_response = legal_action
                break
        if not find_action:
            action_content = 'Task'
            action_response = 'Task'
            reason = "I need to collect more resources for upgrade."
        
        return action_content, action_response, reason

    
    def generate_action(self, inventory, escrow, capability, rseource_maps, agent_locs, market_hist, bargain_message = None, masks = None, nearest = True):
        action_text = None
        # handle messages
        req = {}
        self.agent_loc = agent_locs[self.agent_id]
        resource_obs = self.generate_obs(rseource_maps, self.agent_loc)
        resource_obs_summary = {}
        resource_locs = []
        resource_desps = []
        for resource in resource_obs.keys():
            res_sum_value = sum(resource_obs[resource]['values'])
            resource_obs_summary[resource] = res_sum_value
            resource_locs.extend(resource_obs[resource]['locs'])
            resource_desps.extend(resource_obs[resource]['description'])

        req['inventory'] = { (self.reverse_dict[k] if k in self.reverse_dict.keys() else k):int(v) for k,v in inventory.items()}
        req['escrow'] = { (self.reverse_dict[k] if k in self.reverse_dict.keys() else k):int(v) for k,v in escrow.items() if k in ['Mat', 'Token']}
        req['actions'] = self.action_space #self.masked_actions(req['inventory'], market_hist['available_asks'])
        req['capability'] = capability
        req['resource_obs_summary'] = resource_obs_summary
        req['steps'] =  2 * self.visible_width
        req['sell_num'] = int(np.sum(market_hist['available_asks']))
        req['bid_num'] = int(np.sum(market_hist['available_bids']))
        req['lowest_price'] = self.get_min_price(market_hist['available_asks'])
        req['highest_price'] = self.get_max_price(market_hist['available_bids'])
        req['avg_market_rate'] = None if market_hist['market_rate'] == 0 else round(market_hist['market_rate'],2)
        req['profile'] = self.profile #profiles[self.type + '_profile']
        
        current_obs = self.obs_template.format(**req) 
        current_obs = current_obs if not self.last_illegal_flag else "Your inventory is not enough for applying your chosen action, so nothing happens. Your should explore other actions." + current_obs
        self.last_illegal_flag = False
        history_str = self.form_obs_history(self.obs_history, self.action_history, self.reason_history, current_obs)
        req['history_str'] = history_str
        if not self.task_finish:
            action_content = 'Task'
            action_response = 'Task'
            reason = "I need to collect more resources for upgrade."
        else:
            prompt_str = self.background.format(**req)
            response, actual_prompt = self.get_LLM_response(prompt_str)
            action_content, action_response, reason = self.parse_response(response)
            if action_content == "Task":
                self.task_continue = 0
                self.task_finish = False
            self.obs_history.append(current_obs)
            self.action_history.append(action_response)
            self.reason_history.append(reason)

        if not self.test_enough(action_content, req["inventory"]):
            self.last_illegal_flag = True
            action_content = 0
        if action_content in self.one_step:
            action_text =  action_content
        elif action_content == 'Task': 

            sorted_indexes = sorted(range(len(resource_locs)), key=lambda i: abs(resource_locs[i][0]) + abs(resource_locs[i][1]))[:3]
            task_decision = self.generate_direction(resource_locs[sorted_indexes[0]] if len(sorted_indexes) > 0 else None)

            action_text =  action_content + '+' + task_decision
        elif action_content == 'Shop':
            shop_decision = self.generate_shop_decision(req['inventory'])
            action_text = action_content + '+' + shop_decision
        else:
            action_text = action_content
        
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

        else:
            action_num = int(action_) + 1
        
        return {component_name : action_num}