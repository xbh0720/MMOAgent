from .LLM_chat import llm_server
import re
import random
import logging
import numpy as np 
import json

class Person_Agent:
    def __init__(self, agent_id, player_profile = None) -> None:
        self.action_memory = []
        self.Prompt_Config = {
            "stop": None,
            "temperature": 0.3,
            "maxTokens": 200,
        }
        self.action_space = ['Upgrade', 'Task', 'Shop', 'Recharge', 'Sell', 'Buy', 'Offline'] #['Upgrade', 'Task', 'Shop', 'Recharge', 'Market'] 
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
        self.initial_bargain_content = "Hi, the material's price is {} Tokens."


    def reset_memory(self):
        self.action_memory = []

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

    
    def generate_direction(self):
        operate_str = 'Please enter the corresponding number of direction you move: \n Left : 1, Right : 2, Up : 3, Down : 4\n'
        print(operate_str)
        direction = None
        '''text = widgets.Text()
        display(text)
        def input_text_value(sender):
            direction = sender.value
        text.on_submit(input_text_value)'''
        direction = input('Enter your direction: ')
        while  not direction.isdigit() or not int(direction) in range(1,5):
            #text.on_submit(self.input_text_value)
            direction = input('Enter your direction: ')
        reverse_action = {'1':'Left', '2':'Right', '3':'Up', '4':'Down'}
        return reverse_action[direction]

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
        background = """There are 6 kinds of activities you could act in the game as follows:
            Upgrade: Consuming 1 Experience, 1 Token and 1 Material to improve 10 capability
            Task: Collecting Experience, Token, or Material with labor by moving to the corresponding location in the game map.
            Shop: Consuming 10 Token to buy 1 Experience or 1 Material in the game mall.
            Recharge: Consuming 1 Currency to buy 10 Token when lacking Token.
            Sell: Selling excess Material to other players to obtain Token.
            Buy: Buying Material using Token from the auction which can be cheaper than Shop but should wait for the Seller and bid with other players."""
        print(background)
        obs_str = """ You currently possess the following number of resources: {inventory}. Additionally, you escrow the following number of resources in auction: {escrow}. You current capability is {capability}.
            The number of available selling Material in the auction is {sell_num} and the lowest selling price is {lowest_price}. The highest bidding price at {highest_price} tokens. The past average transaction price in auction is {avg_market_rate}. """
        print(obs_str.format(**req))
        operate_str = "You can choose one activity by enter the corresponding number: \n Task : 1, Shop : 2, Recharge : 3, Upgrade : 4, Buy : 5, Sell : 6 \n"
        print(operate_str)
        action_number = 0
        

        action_number = input('Enter your action:')
        reverse_action = {'1':'Task', '2':'Shop', '3':'Recharge', '4':'Upgrade', '5':'Buy', '6':'Sell'}
        return reverse_action[action_number]
    

    def get_bargain_price(self, opp_agent_id):
        prompt_str = '''Here are two players' bargain dialogue history on game resource Material: {bargain_history}. 
            Question: What price do they both agree with? You should respond the following format:
            Response format:
            Price: a price number in integer
            '''
    
        prompt_config = {
                "stop": None,
                "temperature": 0.3,
                "maxTokens": 20
            }
        req = {'bargain_history':self.to_bargain_history(opp_agent_id)}
        mode = 'chat'
        response, _, actual_prompt = llm_server(
            {"prompt": prompt_str.format(**req), **prompt_config}, mode
        )
        logging.info('Human success price \n' + str(actual_prompt) + '\n' + 'Huamn Response: \n' + response)
    
        price_match = re.search(r'Price:(.*)', response)
        if price_match:
            price = price_match.group(1).lstrip(' ').rstrip('.')
            if price.isdigit():
                return int(price)
        return None
    
    def has_seller(self): 
        for object_id in self.bargain_state_all.keys(): 
                if self.bargain_state_all[object_id]['in_bargain'] == True:
                    return True
        return False

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
        else: 
            for message in messages:
                  
                speaker_id = message['speaker_id']
                if not speaker_id in self.bargain_state_all.keys() or self.bargain_state_all[speaker_id]['in_bargain'] == False:

                    if message['receiver_id'] == 'all':
                        if self.has_bargain_object() != None: # choose only one seller (bargain object) here
                            pass  #continue find object's message
                        else: #one2one
                            print('You receive a message from agent {} saying to all: {} \n'.format(message['speaker_id'], message['content']))
                            print('You should decide whether to chat with him: chat : 0, not chat : 1 \n')
                            decision_number = input('Enter your decision: ')
                            while not int(decision_number) in [0, 1]:
                                decision_number = input('Enter your decision: ')
                            logging.info('Human Buy decision. \\' + str(decision_number))
                            #bargain_state = re.search(r'Bargain statement:(.*)', buy_response).group(1).lstrip(' ').rstrip('\n')
                            if int(decision_number) == 0:                      
                                self.bargain_state_all[speaker_id] = {'in_bargain':True, 'bargain_role':'Buyer', 'bargain_memory':[], 'wait_round':0}
                                print('You should input your response to the seller: ')
                                return_content = input()
                                return_message = self.to_Message(return_content, self.agent_id, speaker_id)
                                self.bargain_state_all[speaker_id]['bargain_memory'] += [message, return_message]
                            else:
                                pass
                    else:  # seller first receive a buy's message
                        if self.has_bargain_object() != None: # choose only one buyer (bargain_objext)
                            pass
                        else:
                            print('You receive a message from agent {} want buy your material: {} \n'.format(message['speaker_id'], message['content']))
                            initial_price = re.search(r'\d+', self.seller_init_message['content']).group()
                            self.bargain_state_all[speaker_id] = {'in_bargain':True, 'bargain_role':'Seller', 'bargain_memory':[self.seller_init_message], 'wait_round':0, 'initial_price':int(initial_price)}
                            print('You should input your response to the buyer: ')
                            return_content = input()
                            return_message = self.to_Message(return_content, self.agent_id, speaker_id)
                            self.bargain_state_all[speaker_id]['bargain_memory'] += [message, return_message]
                        
                else:
                    receiver_id = message['receiver_id']
                    if receiver_id != 'all':
                        if self.bargain_state_all[speaker_id]['bargain_role'] == 'Buyer':
                            if message['content'] == 'The resource has been selt!':
                                self.bargain_state_all[speaker_id] = {'in_bargain':False, 'bargain_role':None, 'bargain_memory':[], 'wait_round':0}
                            else:
                                print('You receive a message from seller agent {} saying: {} \n Your chat history is {}'.format(message['speaker_id'], message['content'], self.to_bargain_history(speaker_id)))
                                print('You should input your response to the seller: ')
                                return_content = input()
                                return_message = self.to_Message(return_content, self.agent_id, speaker_id)
                                self.bargain_state_all[speaker_id]['wait_round'] = 0
                                self.bargain_state_all[speaker_id]['bargain_memory'] += [message, return_message]
                        elif self.bargain_state_all[speaker_id]['bargain_role'] == 'Seller':
                            self.bargain_state_all[speaker_id]['bargain_memory'] += [message]
                            print('You receive a message from buyer agent {} saying: {} \n Your chat history is {}\n'.format(message['speaker_id'], message['content'], self.to_bargain_history(speaker_id)))
                            self.bargain_state_all[speaker_id]['wait_round'] = 0
                            print('Do you agree selling the material? Yes: 1, Continue: 2, Stop: 3 \n')
                            deal_end = input('Enter your choice: ')
                            if int(deal_end) == 2:
                                print('You should input your response to the buyer: ')
                                return_content = input()
                                return_message = self.to_Message(return_content, self.agent_id, speaker_id)
                                self.bargain_state_all[speaker_id]['bargain_memory'] += [return_message]
                            elif int(deal_end) == 1:
                                price = self.get_bargain_price(speaker_id)
                                if price:
                                    return_message = {'new_price': price, 'seller_id': self.agent_id, 'buyer_id': speaker_id}
                                    logging.info('Successful bargain history with human:' + self.to_bargain_history(opp_agent_id=speaker_id))
                                self.bargain_state_all[speaker_id].update({'in_bargain':False, 'bargain_role':None, 'bargain_memory':[], 'wait_round':0})
                            else:
                                self.bargain_state_all[speaker_id].update({'in_bargain':False, 'bargain_role':None, 'bargain_memory':[], 'wait_round':0})
                            
        self.round_expired() 
        if return_message:
            action_text = 'Bargain' + '+' + json.dumps(return_message)
            return action_text
        return None


    def generate_action(self, inventory, escrow, capability, rseource_maps, agent_locs, market_hist, bargain_message = None, nearest = True):
        action_text = None
        # handle messages
        req = {}
        resource_obs = self.generate_obs(rseource_maps, agent_locs[self.agent_id])
        resource_obs_summary = {}
        resource_locs = []
        resource_desps = []
        
        for resource in resource_obs.keys():
            res_sum_value = sum(resource_obs[resource]['values'])
            resource_obs_summary[resource] = res_sum_value
            resource_locs.extend(resource_obs[resource]['locs'])
            resource_desps.extend(resource_obs[resource]['description'])

        req['inventory'] = { (self.reverse_dict[k] if k in self.reverse_dict.keys() else k):v for k,v in inventory.items()}
        req['actions'] = self.action_space#self.masked_actions(req['inventory'], market_hist['available_asks'])
        req['capability'] = capability
        req['escrow'] = { (self.reverse_dict[k] if k in self.reverse_dict.keys() else k):v for k,v in escrow.items() if k in ['Mat', 'Token']}
        req['resource_obs_summary'] = resource_obs_summary
        req['steps'] =  2 * self.visible_width
        req['sell_num'] = np.sum(market_hist['available_asks'])
        req['lowest_price'] = self.get_min_price(market_hist['available_asks'])
        req['highest_price'] = self.get_max_price(market_hist['available_bids'])
        req['avg_market_rate'] = None if market_hist['market_rate'] == 0 else market_hist['market_rate']
        req['profile'] = self.profile #profiles[self.type + '_profile']
        req['action_memory'] = self.action_memory[-5 : ]
        req['online_step'] = self.online_step
        
        action_text = self.on_receive_message(bargain_message, req)

        if action_text == None:
            action_content = self.generate_first_legal_action(req)

            if action_content in self.one_step:
                action_text =  action_content
            elif action_content == 'Task': # left ,right, up, down make sure the number
                task_decision = self.generate_direction()
                action_text =  action_content + '+' + task_decision
            elif action_content == 'Shop':
                actions = ['Material', 'Experience']
                
                shop_decision = self.generate_shop_decision(req['inventory'])
                action_text = action_content + '+' + shop_decision
            
            elif action_content == 'Sell':    
                print('There are two ways for selling the Material. Auction: the selling order will retain for a period of time but should pay tax for the auction\'s service, which is 10 percent of the final deal price. \n Private: Directly sending messages to all palyers expressing ideas for private transaction where you do not need to pay tax but the order only exists for short time.')
                print('You should decide the selling way. Auction: 0, Private: 1.\n And your decided selling price between 1 and 10 Tokens.')
                decison_number = input('Selling way:')
                while not decison_number.isdigit() or not int(decison_number) in [0,1]:
                    decison_number = input('Selling way:')
                price = input('Selling price:')
                while not price.isdigit():
                    price = input('Selling price:')
                if int(decison_number) == 0:
                    action_text = 'Market' + '.' + action_content + '_' + 'Mat' + '+' + str(price)
                else: 
                    object_id = self.has_bargain_object()
                    
                    if object_id != None and self.bargain_state_all[object_id]['bargain_role'] == 'Seller':
                        price = max(int(price) + 1, int(1.25 * int(price))) 
                        action_text = 'Market' + '.' + action_content + '_' + 'Mat' + '+' + str(price) 
                    else:
                        message = self.to_Message(self.initial_bargain_content.format(price), self.agent_id, 'all')
                        action_text = 'Bargain' + '+' + json.dumps(message)
                        self.seller_init_message = message
                
               
            elif action_content == 'Buy':
                print('You should decide your price bidding on the material in Auction between 1 and 10 Tokens..')
                price = input('Bidding price:')
                while not price.isdigit():
                    price = input('Bidding price:')
                action_text = 'Market' + '.' + action_content + '_' + 'Mat' + '+' + str(price)
                
            else:
                action_text = 0
        
        action = self.parse_action(action_text)
        if action == 0:
            self.action_memory.append('NO-OP')
            self.online_step = 0
        else:
            self.action_memory.append(list(action.keys())[0])
            self.online_step += 1
            self.total_online_steps += 1
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