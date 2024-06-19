from .LLM_chat import llm_server
from .A_star import AStar
import re
import random
import logging
import numpy as np 
import json
import math
import time
from collections import deque
from copy import deepcopy
from datetime import datetime

# LLM choose proper action through observation, {subspace：action}

class Long_term_memory:
    def __init__(self) -> None:
        self.memory = []
        self.forget_factor = math.exp(-0.02)
        self.importance_bound = 0.2
        self.max_length = 20

    def add_memory(self, record, importance): # add record to the memory, the similar one will be merged to enhance the importance score
        emb = self.record2emb(record)
        merged = False
        for item in self.memory:
            if self.similarity(emb, item['emb']) > 0.9 and item['record']['action'] == record['action']:
                merged = True
                item['importance'] += importance
        if not merged:
            if len(self.memory) > self.max_length:
                min_importance = min(self.memory, key=lambda x : x['importance'])
                self.memory.remove(min_importance)
            self.memory.append({'emb':emb, 'record':record, 'importance':importance})

    def record2emb(self, record):
        inventory_keys = ['Currency', 'Experience', 'Material', 'Token']
        escrow_keys = ['Material', 'Token']
        resources_keys = ['Experience', 'Material', 'Token']
        auction_keys = ['Sell_num', 'Bid_num']
        inventory_list = [record['state']['Inventory'].get(key, 0) for key in inventory_keys] + [record['state']['Escrow'][key] for key in escrow_keys]
        resources_list = [record['env_state']['Resources'].get(key, 0) for key in resources_keys] + [record['env_state']['Auction'][key] for key in auction_keys]
        return inventory_list + resources_list


    def similarity(self, vector1, vector2):
        normalized_values = [abs(x - y) / max(max(x, y), 1) for x, y in zip(vector1, vector2)]
        distance = max(normalized_values)
        return 1.0 - distance

    def read_memory(self, cur_record): # read most similar and important record as demo
        if not self.memory:
            return None
        key_emb = self.record2emb(cur_record)
        scores = [self.similarity(key_emb, item['emb']) + item['importance'] for item in self.memory]
        max_score_index = scores.index(max(scores))
        record = self.memory[max_score_index]['record']
        return record
    
    def forget(self):
        for item in self.memory[:]:
            if item.get('importance', 0) * self.forget_factor < self.importance_bound:
                self.memory.remove(item)
            item['importance'] *= self.forget_factor


class LLM_Agent2:
    def __init__(self, agent_id, player_profile = None, off_rate = None) -> None:
        self.background = '''You plays a role as a game player. Your profile is: {profile} 
The game encompasses three in-game primary resources: Experience, Token, and Material, which are both obtainable and expendable in the game. Only the Material is tradable among players. You start with a certain amount of out-of-game resource Currency (i.e. money), which can be used for recharging.
There are 6 kinds of activities you could act in the game as follows:
1. Upgrade: Expending 1 Experience, 1 Token, and 1 Material to improve role's capability.
2. Shop: Spending 10 Tokens to purchase 1 Experience or 1 Material from the in-game mall.
3. Recharge: Using 1 Currency to acquire 10 Tokens when lacking token.
4. Sell: Selling Material to obtain Tokens from other players.
5. Buy: Buying Material from auctions using Tokens, potentially cheaper than the Shop but requires waiting for sellers and bidding with other players.
6. Task: Collecting nearby in-game resources with much effort by navigating the game map.
The following is the game state you observe:
Your inventory: {inventory}
Your escrowment in auction: {escrow}, with materials on selling and tokens on bidding. The escrowment resources cannot be used elsewhere. 
Your capability: {capability}
Auction info: {sell_num} materials on selling with the lowest selling price at {lowest_price} tokens. {bid_num} puchasing needs from buyers, with the highest bidding price at {highest_price} tokens. The past average deal price in auction is {avg_market_rate} tokens.
Nearby resources: {resource_obs_summary}. If no resources are nearby, you can also further explore through Task.
Your previous chosen actions: {action_memory}
Your recent reflection: {cycle_reflect}
'''
        self.background2 = '''You plays a role as a game player. Your profile is: {profile}
There are 3 kinds of in-game resources: Experience, Token, and Material, which can be obtained and consumed in the game and are indispensable resources for upgrading game role's capability.
You also have a certain number of out-of-game Currency (i.e. money) which can be used to recharge for Tokens and 1 currency can exchange 10 tokens. 
You currently possess the following number of resources: {inventory}. Additionally, you escrow the following number of resources in auction: {escrow}, with materials for selling and tokens for bidding.
 '''

        self.action_template = '''Now, you should choose proper activity according to the above information and obeys your profile. You should only respond in the format as described below:
Reasoning: Based on the given information, do brief reasoning step by step about what your next action should be.
Action: The next action

Please remember, the action returned must be only one word in {actions} without quotes or any other modifiers. The Reasoning content should be brief in 30 words.
''' 
        self.market_sell_template = '''You choose to sell your material in current step. You have two options for selling your material: through an auction or via private transaction with other players. The auction requires less effort but incurs a 10% tax on the final deal price. Private transactions don't incur tax but demand more effort to communicate with buyers.   
You should first decide which way you choose to sell the Material and then specify a proper price using Token considering the market's supply and demand law.
The average deal price in auction is {avg_market_rate} Tokens. There are currently {sell_num} materials available for auction, with the lowest selling price at {lowest_price} tokens. There are {bid_num} bids from buyers with purchasing needs for Material in auction, with the highest bidding price at {highest_price} tokens. 
You should try to set a price between 1 and {price_level}, balancing competitiveness and profitability, while considering the material's official shop selling price of 10 tokens as a reference. You should also consider the underlying waiting time when setting price, if you could accept and set a price no more than the current highest bidding price in auction, you can sell the material immediately, otherwise you may have to wait until a buyer accepts your price.
You should only respond in the format as described below:
Reasoning: Based on the given information, do brief reasoning step by step about your decision and price.
Decision: The selling way you choose.
Price: The decided price in integer

Please remember, the Decision much be one word in [Auction, Chat], the Price returned must be a integer between 1 and {price_level}, and the Reasoning content should be brief in 20 words.
'''
        self.market_buy_template = '''You choose to buy Material in the auction at current step. The average deal price is {avg_market_rate}. There are currently {sell_num} materials selling up for auction, with the lowest starting price at {lowest_price} tokens. There are {bid_num} bids from buyers with purchasing needs, with the highest bidding price at {highest_price} tokens. 
You should choose buying the Material with proper price using Token considering the market's supply and demand law.
You should decide the price between 1 and {price_level} not exceeding your processed Token's number. The price should balance competitiveness and profitability, while considering the material's official shop selling price of 10 tokens as a reference. You should also consider the underlying waiting time, if you could offer at least the lowest selling price, you can get the material immediately, otherwise you may have to wait until a seller accepts your price.
You should only respond in the format as described below:
Reasoning: Based on the given information, do brief reasoning step by step about what your specified price should be.
Price: The decided price in integer

Please remember, the Price returned must be a integer between 1 and {price_level}, and the Reasoning content should be brief in 20 words.
'''
        self.reflect_template = '''You choose {action} activity in current step. However, it is illegal as {reason}. So, you should regenerate the action you choose until it's legal.
The following is the previous illeagal actions you choose in current step: {illegal_actions}. 
You should only respond in the format as described below:
Reasoning: Based on the given information, do brief reasoning about what your next action should be.
Action: The next action

Please remember, the Action returned must be only one word in {actions} without quotes or any other modifiers, and the Reasoning content should be brief in 20 words.
'''
        
        self.buy_decision = '''There is a player who is selling Material with price: {sell_sentence}. 
You have {Token_num} Tokens. Now, you should decide whether to chat with the seller and buy his Material or not. If you choose to chat with the seller, you could try to bargain with the player by proposing a lower price to see whether the player would agree but may cost much effort on negotiation.
You should only respond in the format as described below:
Reasoning: Based on the given information, do brief reasoning about what your decision should be.
Decision: Your decision to chat or not

Please remember, the Decision returned must be only one word in [Yes, No] where 'Yes' for chat and 'No' for not chat without any other modifiers, and the reasoning should be brief.
'''
        self.trajectory_memory = deque(maxlen=10)
        self.cycle_reflect_memory = []
        self.long_term_memory = Long_term_memory()
        self.Prompt_Config = {
            "stop": None,
            "temperature": 0.3,
            "maxTokens": 500,
        }
        self.action_space = ['Upgrade', 'Shop', 'Recharge', 'Sell', 'Buy', 'Task']
        self.one_step = ['Upgrade', 'Recharge']
        self.visible_width = 5
        self.reverse_dict = {'Exp':'Experience', 'Mat':'Material'}
        self.price_level = 10
        self.profile = player_profile
        self.agent_id = agent_id
        
        self.bargain_state_all = {} 
        self.seller_init_message = None
        self.initial_bargain_content = "Hi, the material's price is {} Tokens."

        self.reflect_cycle = 15 
        
        self.last_direction = None
        self.tax_rate = 0.1
        self.task_continue = 0
        self.task_finish = True
        self.visited_positions = set()

        self.a_star = AStar()
        self.discount = 0.9

        self.clock_step = 0
        self.off_rate = off_rate

    def reset_memory(self):
        self.trajectory_memory = []

    
    def get_center(self, map, agent_loc):
        r, c = agent_loc
        rows, columns = map.shape
        self.map_height, self.map_width = rows, columns
        w = self.visible_width
        left_margin = np.maximum(0, c - w)
        right_margin = np.minimum(columns - 1, c + w)
        up_margin = np.maximum(0, r - w)
        down_margin = np.minimum(rows - 1, r + w)
        locs = []
        relative_locs = []
        values = []
        for i in range(up_margin, down_margin + 1):
            for j in range(left_margin, right_margin + 1):
                 if map[i, j] > 0:
                     # resource, row, column, value
                    loc = (i, j)
                    relative_loc = (i - r, j - c)
                    locs.append(loc)
                    relative_locs.append(relative_loc)
                    values.append(map[i,j])
        
        return  relative_locs, values

    def generate_obs(self, resources_maps, agent_loc):
        resource = resources_maps.keys()
        resource_obs = {}
        for resource, map in resources_maps.items():
            rev_name = self.reverse_dict[resource] if resource in self.reverse_dict.keys() else resource
            locs, values = self.get_center(map, agent_loc)
            if len(values) > 0:    
                resource_obs[rev_name] = {}
                resource_obs[rev_name]['locs'] = locs
                resource_obs[rev_name]['values'] = values
        
        return resource_obs

    def generate_direction(self, nearest_res_loc):
        if not nearest_res_loc:
            # Deep First Search
            self.task_continue += 1
            if self.task_continue >= 5:
                self.task_finish = True
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
            
            relative_r, relative_c = nearest_res_loc
            if abs(relative_r) + abs(relative_c) <= 1:
                self.task_finish = True
            path = self.a_star.astar(self.agent_loc, (self.agent_loc[0] + relative_r, self.agent_loc[1] + relative_c), map=self.map)
            if self.agent_loc != path[0]:
                length = len(path)
                for i in range(length):
                    if path[0] == self.agent_loc:
                        break
                    else:
                        path.pop(0)
            if len(path) <= 1:
                return None 
            next_p = path[1]
            relative_r, relative_c = next_p[0] - self.agent_loc[0], next_p[1] - self.agent_loc[1]
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
        prompt_config = {"model":'llama3'}
        retry = 3
        while retry > 0:
            retry -= 1
            try:
                response, _, actual_prompt = llm_server(
                {"prompt": prompt_str, **prompt_config}, mode = 'chat'
                )
                return response, actual_prompt
            except:
                print('LLM call failed!')
        return "", ""
    
    def parse_masks(self, masks):
        legal_actions = deepcopy(self.action_space)
        if isinstance(masks, dict):
            for action in self.action_space:
                if action in ['Buy', 'Sell']:
                    action_ = 'Market.' + action + '_Mat'
                else:
                    action_ = action
                action_mask = masks.get(action_, None)
                if not action_mask is None:
                    if len(action_mask) == 1:
                        if not action_mask[0]: # illegal
                            legal_actions.remove(action)
                    else:
                        if action == 'Buy':
                            action_mask[0] = 0
                        if not np.any(np.array(action_mask)!=0):
                            legal_actions.remove(action)

        return legal_actions


    def generate_first_legal_action(self, req):
        tolerance = 3
        is_enough = None
        is_legal_format = None
        
        self.reflect_illegal_memory = []
        background_str = self.background.format(**req)
        
        legal_actions = self.parse_masks(self.masks)
        while not is_enough and tolerance > 0:   
            if is_enough == None and is_legal_format == None:
                prompt_str = background_str   
                related_memory = self.long_term_memory.read_memory(req['cur_record'])
                if not related_memory is None:
                    prompt_str += "Your past successful chosen action for upgrading: {}. ".format(related_memory['action'])
                prompt_str += self.action_template.format(**req)
            else:
                reflect_req = {}
                if is_legal_format == False:
                    reflect_req['reason'] = "you violate the following requirements for generating one-word actions in a specific set."
                else:
                    if action_content == 'Buy' and req['inventory']['Token'] > 0 or action_content == 'Sell' and req['inventory']['Material'] > 0:
                        reflect_req['reason'] = "the order you created in auction has exceeded the limit."
                    reflect_req['reason'] = "your possessed resources' number is not enough for {}".format(action_content)
                reflect_req['action'] = action_content
                reflect_req['actions'] = self.action_space
                reflect_req['illegal_actions'] = self.reflect_illegal_memory
                prompt_str = background_str + self.reflect_template.format(**reflect_req)
                self.reflect_illegal_memory.append(action_content)

            response, actual_prompt = self.get_LLM_response(prompt_str)
            tolerance -= 1
            action_match = re.search(r'Action:(.*)', response)
            reason_match = re.search(r'Reasoning:(.*)', response)
            if reason_match:
                reason = reason_match.group(1).lstrip(' ')
            if action_match:
                action_content = action_match.group(1).lstrip(' ').rstrip('.').replace('\'', '')
                if not action_content in self.action_space:
                    logging.error('error: It is not a legal action. \\' + response)
                    is_legal_format = False
                    continue 
            else:
                logging.error('error: Do not find an action in the text. \\' + response)
                action_content = 0
                actual_prompt = ''
                reason = ''
                continue #return 0
            is_legal_format = True

            is_enough = action_content in legal_actions 
            if not is_enough:
                logging.error('error: the inventory is not enough. \\' + response)

        logging.info('id:' + str(self.agent_id) + ' LLM prompt. \\' + str(actual_prompt) + '\n' + 'LLM response. \\' + response)
        

        return actual_prompt, action_content, reason


    def has_bargain_object(self):# role == seller
        for object_id in self.bargain_state_all.keys():
                if self.bargain_state_all[object_id]['in_bargain'] == True:
                    return object_id
        return None
    
    def round_expired(self):
        for key in self.bargain_state_all.keys():
            if self.bargain_state_all[key]['in_bargain'] and self.bargain_state_all[key]['wait_round'] > 1:
                self.bargain_state_all[key] = {'in_bargain':False, 'bargain_role':None, 'bargain_memory':[], 'wait_round':0}
    
    
    def on_receive_message(self, messages, req): 
        for object_id in self.bargain_state_all.keys():
                if self.bargain_state_all[object_id]['in_bargain'] == True:
                    self.bargain_state_all[object_id]['wait_round'] += 1
        return_message = None
        
        if not messages:          
            pass
        else:  # process messages
            for message in messages:
                speaker_id = message['speaker_id']
                if not speaker_id in self.bargain_state_all.keys() or self.bargain_state_all[speaker_id]['in_bargain'] == False:
                    if message['receiver_id'] == 'all':
                        # buyer first receive seller's  to all message
                        # decide buy or not, if not, do other thing else start bargain             
                        if self.has_bargain_object() != None: # choose only one seller (bargain object) here
                            pass  #continue find object's message
                        else: 
                            buy_req = {'sell_sentence':message['content'], 'Token_num':req['inventory']['Token'], 'lowest_price': req['lowest_price'], 'avg_market_rate':req['avg_market_rate']}
                            buy_decision_str = self.background.format(**req) + self.buy_decision.format(**buy_req) #profile
                            buy_response, buy_actual_prompt = self.get_LLM_response(buy_decision_str)
                            decision_match = re.search(r'Decision:(.*)', buy_response)
                            if decision_match:
                                decision = decision_match.group(1).lstrip(' ')
                                logging.info('id:' + str(self.agent_id) + ' Buy decision LLM prompt. \\' + str(buy_actual_prompt) + '\n' + 'Buy decision LLM response. \\' + buy_response)
                            else:
                                decision = ""
                            if 'yes' in decision.lower():                      
                                initial_price = re.search(r'\d+', message['content']).group()
                                self.bargain_state_all[speaker_id] = {'in_bargain':True, 'bargain_role':'Buyer', 'bargain_memory':[], 'wait_round':0, 'initial_price':int(initial_price)}
                                return_message = self.Buyer_bargain_response(message, buy_req['Token_num'], int(initial_price), req['sell_num'], req['lowest_price'], req['avg_market_rate'])

                                break
                            else:
                                pass
                    else:  
                        if self.has_bargain_object() != None: # choose only one buyer (bargain_objext)
                            pass
                        else:
                            
                            initial_price = re.search(r'\d+', self.seller_init_message['content']).group()
                            self.bargain_state_all[speaker_id] = {'in_bargain':True, 'bargain_role':'Seller', 'bargain_memory':[self.seller_init_message], 'wait_round':0, 'initial_price':int(initial_price)}
                            return_message = self.Seller_bargain_response(message, self.bargain_state_all[speaker_id]['initial_price'], req['bid_num'], req['highest_price'])
                        
                else:
                    receiver_id = message['receiver_id']
                    if receiver_id != 'all':
                        if self.bargain_state_all[speaker_id]['bargain_role'] == 'Buyer':
                            if message['content'] == 'I agree to sell!':
                                self.bargain_state_all[speaker_id] = {'in_bargain':False, 'bargain_role':None, 'bargain_memory':[], 'wait_round':0}
                            else:
                                return_message = self.Buyer_bargain_response(message, req['inventory']['Token'], self.bargain_state_all[speaker_id]['initial_price'], req['sell_num'], req['lowest_price'], req['avg_market_rate'])  #如何判断是不是自己一直在谈的object ，由上面的判断条件确认
                        elif self.bargain_state_all[speaker_id]['bargain_role'] == 'Seller':
                            if len(self.bargain_state_all[speaker_id]['bargain_memory']) > 6:
                                logging.info('id:' + str(self.agent_id) + ' Failed bargain history:' + self.to_bargain_history(opp_agent_id=speaker_id))
                                self.bargain_state_all[speaker_id].update({'in_bargain':False, 'bargain_role':None, 'bargain_memory':[], 'wait_round':0})
                            else:
                                return_message = self.Seller_bargain_response(message, self.bargain_state_all[speaker_id]['initial_price'], req['bid_num'], req['highest_price'])

        self.round_expired()  # expired when not receive new message
        if return_message:
            action_text = 'Bargain' + '+' + json.dumps(return_message)
            return action_text
        return None

    def cycle_reflect(self, req):
        prompt = '''You are now a game player in a mmo game. Your profile is: {profile} The game encompasses three primary resources: Experience, Token, and Material, which are both obtainable and expendable. The amount of Currency is initially fixed, used for recharging but not acquirable afterward. 
There are 8 legal kinds of activities in the game as follows: 
1. Upgrade: Expending 1 Experience, 1 Token, and 1 Material to improve role's capability.
2. Shop: Spending 10 Tokens to purchase 1 Experience or 1 Material from the in-game mall.
3. Recharge: Using 1 Currency to acquire 10 Tokens when lacking token.
4. Sell: Selling Material to obtain Tokens from other players.
5. Buy: Buying Material from auctions using Tokens, potentially cheaper than the Shop but requires waiting for sellers and bidding with other players.
6. Bargain: Negotiating with the seller or buyer on the price of materials. 
7. Task: Collecting nearby Experience, Token, or Material with much effort by navigating the game map.
8. Offline: Getting offline if feelling have spent too much time in the game.
The following are your chosen activities in previous 10 steps: {action_memory}. Your current game state consisting of your inventory and capability is {current_state} and the change of your game state from initial to present is {state_change}. Your escrowment in auction changes from {initial_escrow} to {current_escrow} now, with material for selling and token for bidding. 
The number of different resources around you that can be collected by tasks in the game environment changes from {initial_env_state} in initial to {current_env_state} at present. The selling and bidding needs in auction changes from {initial_auction_state} in initial to {current_auction_state} at present.
The following is your game strategy in last period: {last_strategy}. 
You should first judge whether your activities are consistent with your profile, then reflect and assess whether your action trajectories contribute to the upgrading of your capability considering the change of your game and environment state, finally regenerate your prioritized strategy for your future gameplay on upgrading your capability. 
Remember, the reasoning should use second person format focusing on key points without generating unrelated content. The generated strategy should be practical and ensure that the actions chosen in the strategy are reasonable, which means that your inventory should support the cost of the actions and the actions must be consistent with your profile. Please think step by step.'''

        reflect_req = {'profile':req['profile'], 'actions':req['actions'] + ['Bargain'], 'action_memory':req['action_memory']}
        reflect_req.update({'initial_state':self.trajectory_memory[0]['state']['Inventory'], 
                            'current_state':self.trajectory_memory[-1]['state']['Inventory'],
                            'initial_escrow':self.trajectory_memory[0]['state']['Escrow'],
                            'current_escrow':self.trajectory_memory[-1]['state']['Escrow'],
                            'initial_env_state':self.trajectory_memory[0]['env_state']['Resources'],
                            'current_env_state':self.trajectory_memory[-1]['env_state']['Resources'],
                            'initial_auction_state':self.trajectory_memory[0]['env_state']['Auction'],
                            'current_auction_state':self.trajectory_memory[-1]['env_state']['Auction'],
                            'last_strategy':self.cycle_reflect_memory[-1] if len(self.cycle_reflect_memory) > 0 else "",})
        state_change = {}
        for item in reflect_req['current_state'].keys():
            state_change[item] = reflect_req['current_state'][item] - reflect_req['initial_state'][item]
        reflect_req.update({'state_change':state_change})
        prompt_str = prompt.format(**reflect_req)
        reflect_prompt_config = deepcopy(self.Prompt_Config)
        reflect_prompt_config['maxTokens'] = 500
        response, actual_prompt = self.get_LLM_response(prompt_str, reflect_prompt_config)
        if response == "":
            pass
        else:
            logging.info('id:' + str(self.agent_id) + ' LLM cycle reflect prompt: \\'+ str(actual_prompt) + '\n' + 'LLM cycle reflect response: \\' + response)

            summary_prompt = '''Please briefly summarize the following gaming advice on upgrading game role's capability in the second person within 50 words, focusing on the game strategy and not missing the main points or adding any other content or modifiers: "{}"'''
            summary_response, summary_actual_prompt = self.get_LLM_response(summary_prompt.format(response))
            if summary_response == "":
                pass
            else:
                logging.info('id:' + str(self.agent_id) + ' LLM cycle summary reflect prompt: \\'+ str(summary_actual_prompt) + '\n' + 'LLM cycle summary reflect response: \\' + summary_response)
                self.cycle_reflect_memory.append(summary_response)
                self.trajectory_memory.clear() 
        return

    def generate_action(self, inventory, escrow, capability, rseource_maps, agent_locs, market_hist, bargain_message = None, masks = None):
        action_text = None
        # handle messages
        req = {}
        self.agent_loc = tuple(agent_locs[self.agent_id])
        self.masks = masks
        resource_obs = self.generate_obs(rseource_maps, self.agent_loc)
        resource_obs_summary = {}
        resource_locs = []
        self.clock_step += 1

        self.map = rseource_maps.get(list(rseource_maps.keys())[0])
        for resource in resource_obs.keys():
            res_sum_value = sum(resource_obs[resource]['values'])
            resource_obs_summary[resource] = res_sum_value
            resource_locs.extend(resource_obs[resource]['locs'])

        req['inventory'] = { (self.reverse_dict[k] if k in self.reverse_dict.keys() else k):int(v) for k,v in inventory.items()}
        req['escrow'] = { (self.reverse_dict[k] if k in self.reverse_dict.keys() else k):int(v) for k,v in escrow.items() if k in ['Mat', 'Token']}
        req['actions'] = self.action_space 
        req['capability'] = capability
        req['resource_obs_summary'] = resource_obs_summary
        req['steps'] =  2 * self.visible_width
        req['sell_num'] = int(np.sum(market_hist['available_asks']))
        req['bid_num'] = int(np.sum(market_hist['available_bids']))
        req['lowest_price'] = self.get_min_price(market_hist['available_asks'])
        req['highest_price'] = self.get_max_price(market_hist['available_bids'])
        req['avg_market_rate'] = None if market_hist['market_rate'] == 0 else round(market_hist['market_rate'],2)
        req['profile'] = self.profile #profiles[self.type + '_profile']
        req['action_memory'] = [trajectory['action'] for trajectory in self.trajectory_memory]

        state = {}
        inventory_state = deepcopy(req['inventory'])
        inventory_state.update({'Capability': capability})
        state.update({'Inventory':inventory_state, 'Escrow':req['escrow']})
        env_state = {}
        env_state.update({'Resources':deepcopy(req['resource_obs_summary']), 'Auction':{'Sell_num': req['sell_num'], 'Lowest_selling_price':req['lowest_price'], 'Bid_num':req['bid_num'], 'Highest_bidding_price':req['highest_price']}})
        trajectory = {'state': state, 'env_state':env_state}

        req['cur_record'] = trajectory
        
        # forgetting
        self.long_term_memory.forget()
        # Cycle reflect
        if self.clock_step % self.reflect_cycle == 0:
            self.cycle_reflect(req)
            req['action_memory'] = []
        req['cycle_reflect'] = self.cycle_reflect_memory[-1] if len(self.cycle_reflect_memory) > 0 else None

        action_text = self.on_receive_message(bargain_message, req)

        if action_text == None:
            if not self.task_finish:
                action_content = 'Task'
            else:
                if random.random() < self.off_rate:
                    action_content = "Offline"
                else:
                    if len(self.parse_masks(masks)) <= 1:
                        action_content = "Task"
                    else:
                        actual_prompt, action_content, reason = self.generate_first_legal_action(req)
                    if action_content == "Task":
                        self.task_finish = False
                        self.task_continue = 0

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
                req['action'] = action_content
                req['price_level'] = self.price_level
                tolerence = 3
                price = None
                while tolerence > 0:
                    tolerence -= 1
                    market_prompt_str = self.background2.format(**req) + self.market_sell_template.format(**req) #profile
                    market_response, market_actual_prompt = self.get_LLM_response(market_prompt_str)
                    if market_response == "":
                        continue
                    decision = re.search(r'Decision:(.*)', market_response).group(1).lstrip(' ')
                    price_match = re.search(r'Price:(.*)', market_response)
                    if price_match:
                        price = price_match.group(1).lstrip(' ').rstrip('.').rstrip('tokens').rstrip('token').rstrip(' ')
                    else:
                        logging.error('error: Do not find price decision in the text. \\' + market_response) #经常出现推理很长
                        continue
                    if not price.isdigit() or int(price) < 0 or int(price) > self.price_level:
                        logging.error('error: Illegal price level. \\' + market_response)
                        continue
                    else:
                        logging.info('id:' + str(self.agent_id) + ' Market LLM prompt. \\' + str(market_actual_prompt) + '\n' + 'Market LLM response. \\' + market_response)
                        break
                if not price is None:
                    if 'auction' in decision.lower():
                        action_text = 'Market' + '.' + action_content + '_' + 'Mat' + '+' + str(price)
                    else: 
                        object_id = self.has_bargain_object()
                        if object_id != None and self.bargain_state_all[object_id]['bargain_role'] == 'Seller': 
                            price = max(int(price) + 1, int(1.25 * int(price))) 
                            price = min(self.price_level, price)
                            action_text = 'Market' + '.' + action_content + '_' + 'Mat' + '+' + str(price) 
                        else:
                            message = self.to_Message(self.initial_bargain_content.format(price), self.agent_id, 'all')
                            action_text = 'Bargain' + '+' + json.dumps(message)
                            self.seller_init_message = message
                else:
                    action_text = 0
                    
            elif action_content == 'Buy':
                req['action'] = action_content
                req['price_level'] = min(req['inventory']['Token'], self.price_level)
                req['Token_num'] = req['inventory']['Token']
                tolerence = 5
                reflect_str = ""
                price = None
                while tolerence > 0:
                    tolerence -= 1
                    market_prompt_str = self.background2.format(**req) + self.market_buy_template.format(**req) + reflect_str #profile
                    market_response, market_actual_prompt = self.get_LLM_response(market_prompt_str)
                    if market_response == "":
                        continue
                    price_match = re.search(r'Price:(.*)', market_response)
                    if price_match:
                        price = price_match.group(1).lstrip(' ').rstrip('.').rstrip('tokens').rstrip('token').rstrip(' ')
                    else:
                        logging.error('error: Do not find price decision in the text. \\' + market_response) 
                        continue
                    if not price.isdigit() or int(price) < 0 or int(price) > self.price_level:
                        logging.error('error: Illegal price level. \\' + market_response)
                        continue
                    else:
                        logging.info('id:' + str(self.agent_id) + 'Market LLM prompt. \\' + str(market_actual_prompt) + '\n' + 'Market LLM response. \\' + market_response)
                        if req['inventory']['Token'] < int(price):
                            reflect_str = "You decided price in last attempt is {} tokens. However, it is illegal as your possessed token' number is not enough for buying, so you should be cautious of your decided bidding price.".format(price)
                            continue
                        break
                if not price is None:
                    action_text = 'Market' + '.' + action_content + '_' + 'Mat' + '+' + str(price)
                else:
                    action_text = 0
            else:
                action_text = 0
        
        action = self.parse_action(action_text)
        if action == 0:
            trajectory['action'] = 'Offline'
        else:
            action_key = list(action.keys())[0]
            if 'buy' in action_key.lower():
                trajectory['action'] = 'Bidding on Material using {} tokens in auction'.format(action_text.split('+')[1])
            elif 'sell' in action_key.lower():
                trajectory['action'] = 'Sell Material for {} tokens in auction'.format(action_text.split('+')[1])
            else:
                if 'shop' in action_key.lower():
                    trajectory['action'] = 'Buy {} in shop'.format(action_text.split('+')[1])
                else:
                    trajectory['action'] = action_key
    
            if action_key == 'Upgrade':
                importance = 1
                for i in range(1, 4):
                    importance *= self.discount
                    if len(self.trajectory_memory) > i:
                        self.long_term_memory.add_memory(self.trajectory_memory[-i], importance)

        self.trajectory_memory.append(trajectory)
        
        return action
    
    def to_Message(self, content = "", speaker_id = "", receiver_id = ""):
        message  = {}
        message["content"] = content
        message["speaker_id"] = speaker_id
        message["receiver_id"] = receiver_id
        return message


    def to_bargain_history(self, opp_agent_id):
        reverse_roles = {'Buyer':'Seller', "Seller":'Buyer'}
        bargain_role = self.bargain_state_all[opp_agent_id]['bargain_role']
        return "\n".join(
            [
                f"[{reverse_roles[bargain_role]}] speaks: {message['content']}"
                if message['speaker_id'] != self.agent_id
                else f"[{bargain_role}] speaks: {message['content']}"
                for message in self.bargain_state_all[opp_agent_id]['bargain_memory']
            ]
        )
    
    def Buyer_bargain_response(self, opposite_message, Token_num, initial_price, sell_num, lowest_price, avg_market_rate):
        speaker_id = opposite_message['speaker_id']
        self.bargain_state_all[speaker_id]['wait_round'] = 0
        
        bargain_prompt = '''You're a player in an MMO game, aiming to buy game resource Material from another player using game currency Tokens. 
You have {Token_num} Tokens now. Your historical chat history with the seller is : [{chat_history}]. 
Your goal is to buy the Material with as few Tokens as possible without exceeding your inventory. If the seller's price is reasonable and he is firm, you can agree. 
Consider the auction price as a reference and negotiate with the seller based on that. If the seller's price is higher than the auction's lowest price, ask for a reduction stating that you can buy it from the auction instead of buying from him. There are currently {sell_num} materials selling in auction with the lowest selling price at {lowest_price} tokens.
It's worth noting that the maximum value of Material does not exceed 10 Tokens.
Now, decide whether to accept the seller's price and then return a brief natural language response to the seller based on your decision. You must specify your accepted price in your response every time. You should respond as the following format::
Decision: do brief reasoning step by step about your decision on whether accept the seller's price.
Response: your brief response to the seller without any other content or modifiers.
'''

        self.bargain_state_all[speaker_id]['bargain_memory'].append(opposite_message)
        bargain_req = {'chat_history': self.to_bargain_history(speaker_id)}
        bargain_req['Token_num'] = Token_num
        bargain_req['top_price']= min(int(initial_price * 0.8), Token_num)
        bargain_req['initial_price'] = initial_price
        bargain_req['sell_num'] = sell_num
        bargain_req['lowest_price'] = lowest_price
        bargain_req['avg_market_rate'] = avg_market_rate
        bargain_prompt_str = bargain_prompt.format(**bargain_req)

        bargain_response, bargain_actual_prompt = self.get_LLM_response(bargain_prompt_str)
        decision_match = re.search(r'Decision:(.*)', bargain_response)
        if decision_match:
            decision = decision_match.group(1).lstrip(' ')
        
        bargain_response_match = re.search(r'Response:(.*)', bargain_response)
        if bargain_response_match:
            bargain_response = bargain_response_match.group(1).lstrip(' ')
            logging.info('id:' + str(self.agent_id) + ' Buyer Bargain prompt.\\' + str(bargain_actual_prompt) + '\n' + 'Buyer Bargain Response. \\' + bargain_response + '\n')
        else:
            bargain_response = ""
        
        return_msg =  self.to_Message(content=bargain_response, speaker_id=self.agent_id, receiver_id=opposite_message['speaker_id'])
        
        self.bargain_state_all[speaker_id]['bargain_memory'].append(return_msg)
        return return_msg
    
    def Seller_bargain_response(self, opposite_message, initial_price, bid_num, highest_price):
        speaker_id = opposite_message['speaker_id']
        self.bargain_state_all[speaker_id]['wait_round'] = 0
        bargain_prompt = '''You are now playing a role as a game player in a MMO game. You want to sell Material which is important resource in the game to other players to obtain game currency Tokens and now a buyer is bargaining with you.
Your historical chat history with the buyer is : [{chat_history}]. You are the seller now, and your bottom price is {bottom_price} Tokens. Your goal is to sell the Material with a satisfactory price which means sell more tokens, as high as possible.  But, if the buyer's giving price is not less than your bottom price and his attitude is very determined, you can give in. 
Now, you should only return a brief natural language response to the [Buyer] without any other content or modifiers.
'''
        bargain_prompt = '''You are a player in an MMO game selling important game resource Material to obtain game currency Tokens. Your historical chat history with a buyer is : [{chat_history}]. 
You are the seller and your goal is to sell as many Tokens as possible. But if the buyer's price is reasonable and he is firm, you can agree.
You can consider the highest bidding price from buyers with purchasing needs in the auction as a reference and negotiate with the buyer based on that. If the buyer's price is lower than the auction's highest bidding price, ask for a grow stating that you can sell it in the auction instead of directly selling to him. There are {bid_num} bids from buyers in auction, with the highest bidding price at {highest_price} tokens .
It's worth noting that the maximum value of Material does not exceed 10 Tokens.
Now, decide whether to agree to sell the material at the buyer's price and then return a brief natural language response to the buyer based on your decision. You must specify your accepted price in your response every time.  You should respond as the following format::
Reasoning: do brief reasoning step by step about your decision on whether to agree to sell.
Decision: 'yes' for agree while 'no' for disagree
Price: the deal price if the decision is 'yes' else None
Response: your brief natural language response to the buyer based on you decision without any modifiers.'''

        self.bargain_state_all[speaker_id]['bargain_memory'].append(opposite_message)
        bargain_req = {'chat_history': self.to_bargain_history(speaker_id)}
        bargain_req['bottom_price'] = int(initial_price * 0.8)
        bargain_req['highest_price'] = int(highest_price * (1 - self.tax_rate)) if highest_price else None
        bargain_req['bid_num'] = bid_num
        bargain_prompt_str = bargain_prompt.format(**bargain_req)

        bargain_response, bargain_actual_prompt = self.get_LLM_response(bargain_prompt_str)
        logging.info('id:' + str(self.agent_id) + ' Seller Bargain prompt.\\' + str(bargain_actual_prompt) + '\n' + 'Seller Bargain Response. \\' + bargain_response)
        response_match = re.search(r'Response:(.*)', bargain_response)
        if response_match:
            response = response_match.group(1).lstrip(' ')
        else:
            response = ""
        return_msg = self.to_Message(content=response, speaker_id=self.agent_id, receiver_id=opposite_message['speaker_id'])
        self.bargain_state_all[speaker_id]['bargain_memory'].append(return_msg)
        decision_match = re.search(r'Decision:(.*?)Response:', bargain_response, re.DOTALL)
        if decision_match:
            decision = decision_match.group(1).strip()
        else:
            decision = ""
        if 'yes' in decision.lower():
            price = re.search(r'Price:(.*)', bargain_response).group(1).lower().lstrip(' ').rstrip('.').rstrip('tokens').rstrip('token').rstrip(' ')
            if price.isdigit():
                price_message = {'new_price': int(price), 'seller_id': self.agent_id, 'buyer_id': speaker_id}
                return_msg['content'] = 'I agree to sell!'
                logging.info('id:' + str(self.agent_id) + ' Successful bargain history:' + self.to_bargain_history(opp_agent_id=speaker_id))
                return_msg = [return_msg, price_message]
            else:
                return_msg = []
            self.bargain_state_all[speaker_id].update({'in_bargain':False, 'bargain_role':None, 'bargain_memory':[], 'wait_round':0})
            
        
        return return_msg
    
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
            try:
                action_num = int(action_) + 1
            except:
                return 0
        
        return {component_name : action_num}