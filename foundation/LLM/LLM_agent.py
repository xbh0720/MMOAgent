from .LLM_chat import llm_server
import re
import random
import logging
import numpy as np 
import json
# LLM choose proper action through observation, action格式需得和base-agent中parse_action一致，可以是{subspace：action}的形式

#logging.basicConfig(filename='../gpt3-5-testing-sell_v1.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

'''
subspace

Recharge: n_action = 1, 0 no-op 1 recharge  (cost: 1 currenty, income: 10 tokens) 1 labor
Shop: n_action = 2, shop two commodities EXP and MAT, 0 no-op 1: shop EXP  2:shop MAT   (10 tokens, 1 exp or Mat) 1 labor
Upgrade: n_action = 1, 0 no-op 1 upgrade ("Exp": 1 "Mat": 1 "Token": 1, 10 Cap)  1 labor
Task: n_action = 4, 1: left, 2: right, 3: up, 4: down   move labor 1, if new location have resources, collect 

Market: {commodities} * {Buy, Sell} * {n price levels}  n: 1 + max_bid_ask (10) commodities: tradable  subspace format Buy_{commodity_name}, Sell_{}

first choose the component, then choose the action

observation(是否需要区分不同component，比如粗选时看个大概，细选时给定更多细节)

self inventory, nearby resources, market ? (简化，只考虑是否求购或者挂售)

rule: expense for each component

reward/feedback

Resources: Mat,Exp,Currency, Token
Endogenous: Labor, Capability
'''
'''
The game is set in a mystical realm where players can gather materials through tasks or purchase them using tokens from the in-game shop. 
        Tokens are the primary currency and collectable material, which could be obtainable through real-money recharge or tasks. 
        Additionally, players can engage in trading within an auction, where they can bid on or sell resources. 

Please be very specific about what resources you need to collect.

From (0, 0) to (0, 1) means move Right and from (0, 0) to (1, 0) means move Down.

直接由buy或者sell表示是否更好，因为第一步和第二步之间可能会存在矛盾，比如前面说资源不够所以进行market，后面又选择sell
Market: obtaining or selling Material by trading with other players using your specified number of Token where you can buy Material cheaper than Shop or sell excess Material for Token. 

Market: trading with other players using your specified number of Token where you can buy Material cheaper than Shop or sell excess Material for Token. 

You chose Market in current step with the reason: {reason}. The past average transaction price is {avg_market_rate}, you should choose whether to buy or sell Material with proper price using Token.
            It's a 2-dimensinal decision task, you need to first choose action among [Buy, Sell], then decide the price between 1 and {price_level}.
            You should only respond in the format as described below:
            RESPONSE FORMAT:
            Reasoning: Based on the given information, do brief reasoning about what your next action should be.
            Action: The next action
            Price: The decided price

            Please remember, the action returned must be only one word in [Buy, Sell], the price returned must be a numnber between 1 and {price_level}, and the Reasoning content should be brief in 20 words.

            Here's an example response:
            Reasoning: I am lack of Material and the past average bid is 7.5 Token.
            Action: Buy
            Price: 8
            
'''


'''
新增profile   肝氪特性 要不给定忍受的步数 或者周围没有资源持续探索的步骤

要不要给定no-op操作，这样针对既不愿做任务也不愿充值的人有个对照
You are not so willing to cost Currency, you prefer collecting the resources by Task even it will take much labor, and only when it's very difficult (cost too many steps) to obtain resources through Task will you consider recharge for Token.
You are not so willing to cost too much labor to do Task, you prefer recharging when there are no resource nearby.

You prefer to collect resources through Task, even if it requires significant labor, rather than spending in-game currency. You only considers recharging for tokens when it becomes very difficult (requires too many steps) to obtain resources through tasks.

You are not inclined to invest too much time and effort in doing tasks, you prefer to recharge when no resources are nearby.

'''
#针对疯狂氪金的行为，这里给定阈值50，让其进行反省，阻断其生成recharge操作，但感觉这样定的太死了，阻碍了llm的自由性
reflect_recharge_template = '''
    You choose Recharge activity in current step. However, you currently possess {Token} Tokens unused without necessity recharging for more Tokens. You can consider using these Tokens to buy resources if necessary. Now, you should regenerate the action you choose until it's reasonable.
    You should only respond in the format as described below:
    RESPONSE FORMAT:
    Reasoning: Based on the given information, do brief reasoning about what your next action should be.
    Action: The next action

    Please remember, the Action returned must be only one word in {actions} without quotes or any other modifiers, and the Reasoning content should be brief in 20 words.
    '''

#陷入了wait循环 暂时先不考虑reason不然会带偏
legal_format_template = '''
    You choose "{action}" activity in current step. However, it is illegal as you violate the following requirements for generating one-word actions in a specific set. So, you should correct the action you choose to obey the requirements until it's legal.
    You should only respond in the format as described below:
    RESPONSE FORMAT:
    Reasoning: Based on the given information, do brief reasoning about what your next action should be.
    Action: The next action

    Please remember, the Action returned must be only one word in {actions} without quotes or any other modifiers, and the Reasoning content should be brief in 20 words.
    '''
        

#关于疯狂氪金的行为是否也要用reflect 反思一下，比如当token超过50的时候就提醒它token十分充足，没必要氪金
profiles = {'grind_profile': "You lean towards grinding which means you prefer to collect resources through Task, rather than spending in-game currency. You only considers recharging for tokens when it becomes very difficult (requires too many steps) to obtain resources through tasks.\n", \
            'pay_profile': "You lean towards pay-to-win which means you are not inclined to invest too much time and effort in doing tasks, you prefer to recharge for Token then buy required resources for upgrading when no resources are nearby to collect.\n"}

class LLM_Agent:
    def __init__(self, agent_id, player_type = 'grind') -> None:
        #buy和shop有点重复，导致可能会有点混淆
        #需要增加其他agents的位置，考虑agents之间的竞争
        #拍卖行的obs,主要展示available asks，出价的时候参考available bids，但是如果考虑bargain的话，只考虑最低价，但是若是多个agent同时出价，先到先得，没有成功执行的action需不需要记录到memory
        #bargain状态下observation记录来自另一agent的message 
        self.background = '''Background: You are a player in a game where your main goal is trying to upgrade your capability by interacting with the game. 
            There are 4 kinds of resources in the game: Experience, Token, Currency and Material. 
            The Currency is not collectable and is fixed in the initial time which should be consumed carefully. The other three resources can be obtained in the game.
            There are 6 kinds of activities you could act in the game as follows:
            Upgrade: Consuming 1 Experience, 1 Token and 1 Material to improve 10 capability
            Task: Collecting Experience, Token, or Material with labor by moving to the corresponding location in the game map.
            Shop: Consuming 10 Token to buy 1 Experience or 1 Material in the game mall.
            Recharge: Consuming 1 Currency to buy 10 Token when lacking Token.
            Sell: Selling excess Material to other players to obtain Token.
            Buy: Buying Material using Token from other players which can be cheaper than Shop but should wait for the Seller.
              
            You need to choose proper activity according to the observation of the game enviroment and your resource inventory to achieve higher capability. You currently possess the following number of resources: {inventory}. You current capability is {capability}.
            The number of available selling Material from other players is {sell_num} with the lowest price as {lowest_price}. 
            The number of resources can be collected through Task around you within {steps} steps are: {resource_obs_summary}. If there are no resources around you to be collected, you can also choose to go further through Task to explore the underlying resources.
            '''
        self.action_template = '''Now, your task is to choose a specific activity among activities {actions}. 
            You must follow the following criteria:
            1) You should act as a real palyer obeying your profile.
            2) Your chosen activity's consuming resources' number should be less than your possessed. e.g. only if you have at least 10 Token can you choose Shop.
            3) You should remember your main goal is to achieve higher capability through Upgrade.
            4) You should only respond in the format as described below:
            RESPONSE FORMAT:
            Reasoning: Based on the given information, do brief reasoning about what your next action should be.
            Action: The next action

            Please remember, the action returned must be only one word in {actions} without any other modifiers, and the Reasoning content should be brief in 20 words.

            Here's an example response:
            Reasoning: My inventory is enough to make upgrage.
            Action: Upgrade           
            '''
        self.action_template1 = '''Now, your task is to choose a specific activity among the following current legal activities {actions}. The activities excluded are illegal now as you can not afford the consumming resources.
            You must follow the following criteria:
            1) You should act as a real palyer obeying your profile.
            2) You should remember your main goal is to achieve higher capability through Upgrade.
            3) You should only respond in the format as described below:
            RESPONSE FORMAT:
            Reasoning: Based on the given information, do brief reasoning about what your next action should be.
            Action: The next action

            Please remember, the action returned must be only one word in {actions} without any other modifiers, and the Reasoning content should be brief in 20 words.

            Here's an example response:
            Reasoning: My inventory is enough to make upgrage.
            Action: Upgrade           
            '''
        self.task_template = '''You choose Task in current step with the reason: {reason}. 
            The game map laies out as a {map_height} * {map_width} grid with (0, 0) being at the top-left corner of the grid. You are currently at coordinates {location}. 
            The nearest 3 resources' relative locations from near to far to you are as follows: {resource_obs}. In first dimension of relative location, a negative value means above your position while positive value means below. In second dimension, a negative value means left while positive means right. You should decide which derection to go next among [Left, Right, Up, Down].
            You should only respond in the format as described below:
            RESPONSE FORMAT:
            Reasoning: Based on the given information, do brief reasoning about what your next action should be.
            Action: The next action

            Please remember, the action returned must be one word in [Left, Right, Up, Down], and the Reasoning content should be brief in 20 words. 
            
            Here's an example response:
            Reasoning: There is 1 Token below my location.
            Action: Down
            '''
        self.shop_template = '''You choose Shop activity in current step with the reason: {reason}, you should further choose which resource to purchase between Experience and Material.
            You should only respond in the format as described below:
            RESPONSE FORMAT:
            Reasoning: Based on the given information, do brief reasoning about what your next action should be.
            Action: The next action

            Please remember, the action returned must be one word in [Experience, Material], and the Reasoning content should be brief in 20 words. 
            
            Here's an example response:
            Reasoning: I am lack of Material.
            Action: Material
            '''
        #关于出价太低的问题，后续让两个LLM直接bargain，是否在一个step中完成，还是一个step完成一句话？在环境中如何控制，market step，如何匹配，匹配之后怎么交流（进入某个状态，在该状态下只能进行交流，直到交易完成？），
        #先有seller，再有buyer（只有有挂售的信息，LLM才会选择去buy，进而bargain，比如选择buy之后还要生成砍价的proposal，比如问7元行不行？交由环境发送给另一方，关键是信息怎么传递给env.agent目前只能parse action），作为额外参数放到env.step中？
        #如果是多轮，那么每一轮所对应step的action标记又该是什么‘bargain？’，如果生成buy之后可以接受价格，直接交易成交，如果不能接受，进入bargain（self.next_step）
        #The price should not be too low as you may wait for a long time for a seller accept that price.
        #If you are buying Material, the decided price should not exceed your possessed number of Token.
        #with the reason: {reason}
        self.market_template = '''You choose {action} in current step. The past average transaction price is {avg_market_rate} and the current average selling price of other players in the auction is {avg_sell_price}, you should choose {action} the Material with proper price using Token.
            You should decide the price between 1 and {price_level}. You must try to set a competitive price to but also not sacrifice too much of your own interests to ensure profitability
            
            You should only respond in the format as described below:
            RESPONSE FORMAT:
            Reasoning: Based on the given information, do brief reasoning about what your specified price should be.
            Price: The decided price

            Please remember, the Price returned must be a number between 1 and {price_level}, and the Reasoning content should be brief in 20 words.

            Here's an example response:
            Reasoning: I have enough Material and the past average selling price is 7.5 Token.
            Price: 7
            '''
       
        self.reflect_template = '''
            You choose {action} activity in current step. However, it is illegal as your possessed resources' number is not enough for {action}. So, you should regenerate the action you choose until it's legal.
            The following is the previous illeagal actions you choose in current step: {illegal_actions}. 
            You should only respond in the format as described below:
            RESPONSE FORMAT:
            Reasoning: Based on the given information, do brief reasoning about what your next action should be.
            Action: The next action

            Please remember, the Action returned must be only one word in {actions} without any other modifiers, and the Reasoning content should be brief in 20 words.

            Here's an example response:
            Reasoning: My inventory is enough to make upgrage.
            Action: Upgrade      
            '''
        self.reflect_buy_template = '''
            You choose Buy activity in current step. However, it is illegal as there are no Material available from other players to buy. So, you should regenerate the action you choose until it's legal.
            The following is the previous illeagal actions you choose in current step: {illegal_actions}. 
            You should only respond in the format as described below:
            RESPONSE FORMAT:
            Reasoning: Based on the given information, do brief reasoning about what your next action should be.
            Action: The next action

            Please remember, the Action returned must be only one word in {actions} without any other modifiers, and the Reasoning content should be brief in 20 words.

            Here's an example response:
            Reasoning: My inventory is enough to make upgrage.
            Action: Upgrade      
            '''
        self.cannot_buy_template = '''
            You choose Buy activity in current step. However, it is illegal as you are bargaining with another player and wait his response. So, you should regenerate the action you choose until it's legal.
            The following is the previous illeagal actions you choose in current step: {illegal_actions}. 
            You should only respond in the format as described below:
            RESPONSE FORMAT:
            Reasoning: Based on the given information, do brief reasoning about what your next action should be.
            Action: The next action

            Please remember, the Action returned must be only one word in {actions} without any other modifiers, and the Reasoning content should be brief in 20 words.

            Here's an example response:
            Reasoning: My inventory is enough to make upgrage.
            Action: Upgrade      
            '''
        #针对疯狂氪金的行为，这里给定阈值50，让其进行反省，阻断其生成recharge操作，但感觉这样定的太死了，阻碍了llm的自由性
        self.reflect_recharge_template = '''
            You choose Recharge activity in current step. However, you currently possess {Token} Tokens unused without necessity recharging for more Tokens. You can consider using these Tokens to buy resources if necessary. Now, you should regenerate the action you choose until it's reasonable.
            You should only respond in the format as described below:
            RESPONSE FORMAT:
            Reasoning: Based on the given information, do brief reasoning about what your next action should be.
            Action: The next action

            Please remember, the Action returned must be only one word in {actions} without any other modifiers, and the Reasoning content should be brief in 20 words.

            Here's an example response:
            Reasoning: My inventory is enough to make upgrage.
            Action: Upgrade      
            '''
        
        #陷入了wait循环 暂时先不考虑reason不然会带偏
        self.legal_format_template = '''
            You choose "{action}" activity in current step. However, it is illegal as you violate the following requirements for generating one-word actions in a specific set. So, you should correct the action you choose to obey the requirements until it's legal.
            
            You should only respond in the format as described below:
            RESPONSE FORMAT:
            Reasoning: Based on the given information, do brief reasoning about what your next action should be.
            Action: The next action

            Please remember, the Action returned must be only one word in {actions} without any other modifiers, and the Reasoning content should be brief in 20 words.

            Here's an example response:
            Reasoning: My inventory is enough to make upgrage.
            Action: Upgrade      
            '''
        
        self.buy_template = '''You choose to buy material from another player in the auction, the current lowest selling price is {lowest_price} Token given by another player. 
            You have {Token_num} Tokens. Now, you should decide whether to accept the price or not. If you accept the price, the transaction completes immediately. If you think the price is expensive, you could try to bargain with the player by proposing a lower price to see whether the player would agree.
            You should only respond in the format as described below:
            RESPONSE FORMAT:
            Decision: Your decision
            Bargain statement: Your bargain statement to the other side's player.
            
            Please remember, the Decision returned must be only one word in ['Accept', 'Reject'] without any other modifiers, and the Bargain statement is returned only when the decision is Reject. In the bargain statement, you should try to commuicate with the seller to buy the material with a low price, as low as possible. The statement should be brief.

            Here's an example response:
            Decsion: Reject
            Bargain statement: It's a little expensive, would you consider selling it for 6 Tokens?
            '''
        self.memory = []
        self.Prompt_Config = {
            "stop": None,
            "temperature": 0.3,
            "maxTokens": 80, #总是不完整
        }
        self.action_space = ['Upgrade', 'Task', 'Shop', 'Recharge', 'Sell', 'Buy'] #['Upgrade', 'Task', 'Shop', 'Recharge', 'Market'] 
        self.one_step = ['Upgrade', 'Recharge']
        self.visible_width = 3
        self.reverse_dict = {'Exp':'Experience', 'Mat':'Material'}
        #关于market的cost约束得想想，起码得有TOken进行buy，有mat进行sell，但是第一步不涉及到具体buy还是sell，两者满足其一，第二步进行具体数值比较
        self.costs = {'Upgrade':{'Experience':1, 'Material':1, 'Token':1}, 'Shop':{'Token':10}, 'Recharge':{'Currency':1}, 'Buy':{'Token':1}, 'Sell':{'Material':1}} #'Market':{'Buy':{'Token':1}, 'Sell':{'Material':1}}
        self.price_level = 11
        self.type = player_type
        self.agent_id = agent_id
        '''self.in_bargain = False #标记是不是在bargain，如果是bargain需要持续对话
        self.bargain_role = None
        self.bargain_memory = []
        self.bargain_object = None'''
        self.bargain_state = {'in_bargain':False, 'bargain_role':None, 'bargain_memory':[], 'bargain_object':None, 'initial_price':None, 'wait_round':0}
        self.bargain_state_all = []  #维持所有其他agent的通讯频道，是否要保留信息
        self.min_ask_price = 100
        self.initial_bargain_content = "Hi, the material's price is {} Tokens."


    def reset_memory(self):
        self.memory = []

    #观察一下以agent为中心的（2 * w + 1）范围内的resource，但是这样可能会导致看到的很多，是否应该只考虑nearest的那些，但是这样可能会受离群点影响，比如一个稍微远一点的，但是附近的资源非常集中
    #当agent周围没有物体时又该如何，是随机探索还是给定最近的点（尽管不在视野范围内）
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
        #sorted_indexes = sorted(range(len(relative_locs)), key=lambda i: abs(relative_locs[i][0]) + abs(relative_locs[i][1]))[:3]
        #return [descriptions[i] for i in sorted_indexes], [relative_locs[i] for i in sorted_indexes], [values[i] for i in sorted_indexes]
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
            return random.choice(["Left", "Right", "Up", "Down"])
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
    # 效果非常差
    def masked_actions(self, inventory, available_asks):
        #self.costs['Buy']['Token'] = self.get_min_price(available_asks) ###只有当已有Token超过最低价时可以buy，但是会存在一个bargain的空间？？？
        legal_actions = []
        for action in self.action_space:
            if self.enough_inventory(inventory, action):
                if action == 'Buy':
                    if np.sum(available_asks) > 0:
                        legal_actions.append(action)
                else:
                    legal_actions.append(action)
        return legal_actions
    
    def generate_first_legal_action(self, req):
        # 如果是inbargain，不能产生buy以免套娃
        tolerance = 10
        is_enough = None
        is_legal_format = None
        over_tokens = None
        can_buy = None
        #reflect， 但是会存在一个问题，就是LLM对于多个action都是不合法的时候，可能会出现来回打转的情况，或者一个一个action试过去，这可能需要引入memory, 或者直接在第一次生成的时候就限定masked actions
        #关于memory，一个需要记录历史真实action，一个需要记录reflect中的action
        #action_content 经常是一句话，而不是单纯的一个动作
        self.reflect_illegal_memory = []
        while not is_enough and tolerance > 0:
            if is_enough == None and is_legal_format == None: ###额外加上改正生成格式的步骤
                prompt_str = self.background.format(**req) + profiles[self.type + '_profile'] + self.action_template.format(**req)   #加上profile, 要不要在生成的时候给的action候选集合就是masked掉的
            elif is_legal_format == False:
                format_req = {'action':action_content, 'reason': reason, 'actions':self.action_space}
                prompt_str = self.background.format(**req) + profiles[self.type + '_profile'] + self.legal_format_template.format(**format_req) #加上profile
            elif can_buy == False:
                cannot_buy_req = {'actions':self.action_space, 'illegal_actions':self.reflect_illegal_memory}
                prompt_str = self.background.format(**req) + profiles[self.type + '_profile'] + self.cannot_buy_template.format(**cannot_buy_req) 
            elif over_tokens:
                recharge_reflect_req = {'Token': req['inventory']['Token'], 'actions':self.action_space}
                prompt_str = self.background.format(**req) + profiles[self.type + '_profile'] + self.reflect_recharge_template.format(**recharge_reflect_req) #加上profile
                self.reflect_illegal_memory.append(action_content)
            else:
                reflect_req = {}
                reflect_req['action'] = action_content
                reflect_req['actions'] = self.action_space
                reflect_req['illegal_actions'] = self.reflect_illegal_memory
                if action_content == 'Buy' and req['sell_num'] < 1:
                    prompt_str = self.background.format(**req) + profiles[self.type + '_profile'] + self.reflect_buy_template.format(**reflect_req)  #加profile
                else:
                    prompt_str = self.background.format(**req) + profiles[self.type + '_profile'] + self.reflect_template.format(**reflect_req)  #加profile
                self.reflect_illegal_memory.append(action_content)

            response, _, actual_prompt = llm_server(
            {"prompt": prompt_str, **self.Prompt_Config}, mode = 'chat'
            )
            tolerance -= 1
            #print(response)
            action_match = re.search(r'Action:(.*)', response)
            reason = re.search(r'Reasoning:(.*)', response).group(1).lstrip(' ')
            if action_match:
                action_content = action_match.group(1).lstrip(' ').rstrip('.')
                #print(action_content)
                if not action_content in self.action_space:
                    logging.error('error: It is not a legal action. \\' + response)
                    is_legal_format = False
                    continue #return 0 #当生成的action不在action space中又该如何处理，是循环生成还是后处理,循环生成可能都是一样的
            else:
                logging.error('error: Do not find an action in the text. \\' + response)
                continue #return 0
            is_legal_format = True
            
            if action_content == 'Recharge' and req['inventory']['Token'] > 50:
                over_tokens = True
                continue
            #seller 等buyer的时候也不能buy，只能是seller和buyer二者之一不能同时，如果要同时可以维护两个state，
            #1 buyer对多seller？维护多个speaker的对话历史，但是bargain_state难以维护roles * agents *(in_bargain, memory)
            #如果是多对多场景，类似于拍卖，buyer可以挑便宜的，seller可以挑价高的，但是这个很复杂，类似于多智能体博弈，以及是同时回复所有人还是有顺序之分。。。先搞单线的
            #如果多对多不约束，会出现向已建立联系的买家再次请求buy，自身
            if action_content == 'Buy' and self.bargain_state['in_bargain'] == True: 
                can_buy = False
                continue
            over_tokens = False
            can_buy = True
            is_enough = self.enough_inventory(req['inventory'], action_content)
            if is_enough and action_content == 'Buy':
                if req['sell_num'] < 1:
                    is_enough = False

        #masked actions 
        '''while tolerance > 0:
            if is_legal_format == None: ###额外加上改正生成格式的步骤
                prompt_str = self.background.format(**req) + profiles[self.type + '_profile'] + self.action_template.format(**req)   #加上profile, 要不要在生成的时候给的action候选集合就是masked掉的
            elif is_legal_format == False:
                format_req = {'action':action_content, 'reason': reason, 'actions':self.action_space}
                prompt_str = self.background.format(**req) + profiles[self.type + '_profile'] + self.legal_format_template.format(**format_req) #加上profile
            elif is_legal_format and over_tokens:
                recharge_reflect_req = {'Token': req['inventory']['Token'], 'actions':self.action_space}
                prompt_str = self.background.format(**req) + profiles[self.type + '_profile'] + self.reflect_recharge_template.format(**recharge_reflect_req) #加上profile
                self.reflect_illegal_memory.append(action_content)

            response, _, actual_prompt = llm_server(
            {"prompt": prompt_str, **self.Prompt_Config}, mode = 'chat'
            )
            tolerance -= 1
            #print(response)
            action_match = re.search(r'Action:(.*)', response)
            reason = re.search(r'Reasoning:(.*)', response).group(1).lstrip(' ')
            if action_match:
                action_content = action_match.group(1).lstrip(' ').rstrip('.')
                #print(action_content)
                if not action_content in req['actions']:
                    logging.error('error: It is not a legal action. \\' + response)
                    is_legal_format = False
                    continue #return 0 #当生成的action不在action space中又该如何处理，是循环生成还是后处理,循环生成可能都是一样的
            else:
                logging.error('error: Do not find an action in the text. \\' + response)
                continue #return 0
            is_legal_format = True
            
            if action_content == 'Recharge' and req['inventory']['Token'] > 50:
                over_tokens = True
                continue
            over_tokens = False
            break
        '''
        ### Testing!
        #logging.info('LLM prompt. \\' + str(actual_prompt) + '\n' + 'LLM response. \\' + response)
        
        ### Testing
        if req['inventory']['Material'] > 2:
            return actual_prompt, 'Sell', reason


        return actual_prompt, action_content, reason
    
    def get_min_price(self, price_hist):
        for i in range(np.shape(price_hist)[0]):
            if price_hist[i] > 0:
                return i
        return len(price_hist)
    
    def get_avg_price(self, price_hist):
        prices = np.arange(0, np.shape(price_hist)[0])
        avg_price = prices.dot(price_hist) / np.maximum(
                0.001, np.sum(price_hist)
            )
        return avg_price
    
    def get_seller_id(self, all_asks_hist, lowest_price):
        for agent_id, ask_hist in all_asks_hist.items():
            if agent_id != self.agent_id:
                if ask_hist[lowest_price] > 0:
                    return agent_id
        return None
    
    # 在多对多场景下，每一对seller和buyer的角色不能颠倒，对于一对玩家来说，两者的角色是固定的，但如果涉及到多种物品交易可能会有所倒转，
    # 是否需要考虑以propose_trade的方式，确定每一方付出什么
    def one2one_bargain(self, one_bargain_message):  # baragain_states: speakers * {in_bargain, bargain_memory}
        object_id = one_bargain_message['speaker_id']
        pass


    def is_bargain_end(self):
        prompt_str = '''Here are two players' bargain dialogue history on game resource Material: {bargain_history}. 
            Question: Have the both sides achieved a deal and what price do they both agree with if they achieve the deal? You should respond the following format:
            Response format:
            deal_result: Yes or No
            Price: a price number if deal_result is Yes else be None
            '''
    
        prompt_config = {
                "stop": None,
                "temperature": 0.3,
                "maxTokens": 20
            }
        req = {'bargain_history':self.to_bargain_history()}
        mode = 'chat'
        response, _, actual_prompt = llm_server(
            {"prompt": prompt_str.format(**req), **prompt_config}, mode
        )
        logging.info('Deal End Prompt \n' + str(actual_prompt) + '\n' + 'Deal End Response: \n' + response)
        if 'no' in response.lower():
            return False, 0
        else:
            price_match = re.search(r'Price:(.*)', response)
            if price_match:
                price = price_match.group(1).lstrip(' ').rstrip('.')
                if price.isdigit():
                    return True, price
            return True, self.bargain_state['initial_price']
    
    def get_object_message(self, opposite_message): # 也可能是正在交易的对方关闭了bargain然后又收到了其他message，而且同一个agent是有可能既担任seller又担任buyer，不能确定收到的消息一定是object的
        if not opposite_message:
            return None
        if self.bargain_state['bargain_object'] == None:
            return opposite_message[0]
        for message in opposite_message:
            if message['speaker_id'] == self.bargain_state['bargain_object']:
                return message
        return None


    def generate_action(self, inventory, capability, rseource_maps, agent_locs, market_hist, bargain_message = None, nearest = True):
        #应该要双方建立起联系之后再确定进入bargain，否则会出现多个agent同时给一个seller发送消息（是否考虑并发？）暂时考虑seller只绑定一个buyer进行bargain
        #关于match里不一定匹配，应该是先到先得，谁先确定buy谁和最便宜的进行匹配
        self_min_ask_price = self.get_min_price(market_hist['my_asks'])
        bargain_message = self.get_object_message(bargain_message)
        action_text = None
        if bargain_message == None:
            self.bargain_state['wait_round'] += 1
        #self.get_seller_id(market_hist['all_asks_hists'], self.bargain_state['initial_price']) != None
        #later bargain rounds
        if self.bargain_state['in_bargain'] and bargain_message != None:  #这一步就相当于是绑定操作，因为后续没有收到消息的话直接退出bargain说明没有绑定，但是会存在后续bargain过程中另一个agent给seller发信，可能会占据掉原本agent的位置，需要让seller判断object
            #需要先确认是否end，如果end直接生成buy操作进行购买
            #bargain的过程中物品被直接买走了怎么办？check一下？
            #self.bargain_state['bargain_memory'].append(bargain_message)
           
            #判断东西是不是卖出去了
            ###### 忽略了轮次，这会导致同时说话！！！！！seller和buyer是交替着来的，当seller准备回话的时候，buyer是wait还是做其他？万一产生了新的buy
            if self.bargain_state['bargain_role'] == 'Buyer':
                if market_hist['all_ask_hists'][bargain_message['speaker_id']][self.bargain_state['initial_price']] > 0: 
                    resp_message = self.Buyer_bargain_response(bargain_message, inventory['Token'], self.bargain_state['initial_price'])
                else:
                    resp_message = None
            else:
                if market_hist['my_asks'][self.bargain_state['initial_price']] > 0:
                    resp_message = self.Seller_bargain_response(bargain_message, self.bargain_state['initial_price'])
                else:
                    resp_message = None
            # send message
            if resp_message:
                #self.bargain_state['bargain_memory'].append(resp_message)
                deal_end, price = self.is_bargain_end()
                if deal_end:
                    #如果是subcomponent，也行obs放上了所有agents ask的记录，seller改价，依然用bargain只是在action内容中标记price表明结束了，进而改价
                    #所以这里不用简单的buy进行操作，依然是bargain
                    deal_message = {'old_price':self.bargain_state['initial_price'], 'new_price': price, 'seller_id': self.agent_idx if self.bargain_state['bargain_role'] == 'Seller' else bargain_message['speaker_id']}
                    action_text =  'Market.Bargain_Mat' + '+' + json.dumps(deal_message)  #正常是message格式，这里仅仅是price标识是否end？
                    self.bargain_state.update({'in_bargain':False, 'bargain_role':None, 'bargain_memory':[], 'bargain_object': None, 'initial_price':None})
            
                elif len(self.bargain_state['bargain_memory']) > 10:
                    pass
                    #交易失败无需回复
                else:
                    action_text =  'Market.Bargain_Mat' + '+' + json.dumps(resp_message)
        
        elif not self.bargain_state['in_bargain'] and bargain_message != None:  #be the seller first time
            #seller是从buyer那得到initial price还是以自身当前最小的price为。。可能会出现已经卖掉的情况
            if self.min_ask_price < self_min_ask_price: #been selt
                pass
            else:
                # add initial messages
                self.bargain_state['bargain_memory'] = [self.to_Message(self.initial_bargain_content.format(self_min_ask_price), speaker_id=self.agent_id, receiver_id=bargain_message['speaker_id'])]

                self.bargain_state.update({'in_bargain':True, 'bargain_role':'Seller', 'bargain_object': bargain_message['speaker_id'], 'initial_price':self_min_ask_price})

                resp_message = self.Seller_bargain_response(bargain_message, self.bargain_state['initial_price'])
                action_text =  'Market.Bargain_Mat' + '+' + json.dumps(resp_message)
        
        if action_text == None:
            # 超过一定steps没有回复就退出，无需发一个finish信号
            if self.bargain_state['in_bargain']:  #不一定message非得是none，东西卖出去了或者交易失败无action_text,也同样退出bargain，未手动
                if self.bargain_state['wait_round'] <=1 and bargain_message == None:
                    pass
                else: #been selt or fail deal
                    self.bargain_state.update({'in_bargain':False, 'bargain_role':None, 'bargain_memory':[], 'bargain_object': None, 'initial_price':None})
                #不一定一个step不发消息就代表结束了，但是间断的话万一发起新的buy操作就套娃了
            
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
            req['resource_obs_summary'] = resource_obs_summary
            req['steps'] =  2 * self.visible_width
            req['sell_num'] = np.sum(market_hist['available_asks'])
            req['lowest_price'] = self.get_min_price(market_hist['available_asks']) if req['sell_num'] > 0 else None
            
            
            actual_prompt, action_content, reason = self.generate_first_legal_action(req)

            if action_content in self.one_step:
                action_text =  action_content
            elif action_content == 'Task': # left ,right, up, down make sure the number

                sorted_indexes = sorted(range(len(resource_locs)), key=lambda i: abs(resource_locs[i][0]) + abs(resource_locs[i][1]))[:3]
                task_decision = self.generate_direction(resource_locs[sorted_indexes[0]] if len(sorted_indexes) > 0 else None)

                action_text =  action_content + '+' + task_decision
            elif action_content == 'Shop':
                actions = ['Material', 'Experience']
                
                shop_decision = self.generate_shop_decision(req['inventory'])
                action_text = action_content + '+' + shop_decision
            #obs可以观察到available asks以及available bids、self_asks、以及self_bids
            #sell 也同样可以选择在公聊中叫卖还是挂在auction中（优先级？）
            elif action_content == 'Sell':    #in ['Buy', 'Sell']: #== 'Market': #action_space_name: Market.Buy_Mat  经常出现buy的token不够，sell没有mat，且token比较多的前面的求购还没实现就一直buy
                #actions = ['Buy', 'Sell']
                #这部分有一个问题就是出价出的很低一般就一两个token，因为给定的参考平均价格太低了，一开始是0，一旦有一个就会拉低
                market_req = {}
                #market_req['reason'] = reason
                market_req['action'] = action_content
                market_req['avg_market_rate'] = market_hist['market_rate']
                market_req['price_level'] = self.price_level
                market_req['avg_sell_price'] = self.get_avg_price(market_hist['available_asks'])
                while True:
                    market_prompt_str = self.background.format(**req) + profiles[self.type + '_profile'] + self.market_template.format(**market_req) #profile
                    market_response, _, market_actual_prompt = llm_server(
                        {"prompt": market_prompt_str, **self.Prompt_Config}, mode = 'chat'
                    )
                    price_match = re.search(r'Price:(.*)', market_response)
                    if price_match:
                        price = price_match.group(1).lstrip(' ').rstrip('.')
                    else:
                        logging.error('error: Do not find price decision in the text. \\' + market_response) #经常出现推理很长
                        continue
                    if not price.isdigit() or int(price) < 0 or int(price) > self.price_level:
                        logging.error('error: Illegal price level. \\' + market_response)
                        continue
                    else:
                        logging.info('Market LLM prompt. \\' + str(market_actual_prompt) + '\n' + 'Market LLM response. \\' + market_response)
                        '''if action_content == 'Buy':   #这里改成了sell，需要在buy里面考虑，如果只参考最低价，和最低价比较
                            if req['inventory']['Token'] < int(price):
                                continue'''
                        break
                action_text = 'Market' + '.' + action_content + '_' + 'Mat' + '+' + str(price)
            #这里的buy比较多样，可以考虑是在拍卖行通过挂单的形式进行求购（后续供需关系的计算），还是向公聊中售卖的seller进行私聊
            #当没有玩家叫卖的时候，当然选择挂单，当有玩家叫卖的时候，可以选择发起私聊，那以前的挂单怎么处理？
            #为售卖加税，在bargain时以auction成交的价格作为价格区间，这也可以作为一开始要不要讨价还价的理由
            #那这样必须把bargain的部分拆出来，因为和auction不是一个流程
            elif action_content == 'Buy':
                buy_req = {}
                #####Testing!
                buy_req['lowest_price'] = req['lowest_price'] #1
                buy_req['Token_num'] = req['inventory']['Token']
                seller_id = self.get_seller_id(market_hist['all_ask_hists'], buy_req['lowest_price']) #can not be None, as constrained in generate Buy
                initial_message = self.to_Message(self.initial_bargain_content.format(buy_req['lowest_price']), seller_id, self.agent_id)
                #######Testing!
                #seller_id = 0
                if buy_req['Token_num'] < buy_req['lowest_price']: ###可能会出现bargain到最后钱也不够的情况
                    #直接进入bargain，无需decision，如果相差金额太多该如何

                    self.bargain_state.update({'in_bargain':True, 'bargain_role':'Buyer', 'bargain_object': seller_id, 'initial_price':buy_req['lowest_price'], 'bargain_memory':[]})#由于initial message已经作为opp msg，无需多加
                    bargain_statement = self.Buyer_bargain_response(initial_message, buy_req['Token_num'], buy_req['lowest_price'])
                    action_text = 'Market.Bargain_Mat' + '+' + json.dumps(bargain_statement)
                else:
                    buy_prompt_str = self.background.format(**req) + profiles[self.type + '_profile'] + self.buy_template.format(**buy_req) #profile
                    buy_response, _, buy_actual_prompt = llm_server(
                        {"prompt": buy_prompt_str, **self.Prompt_Config}, mode = 'chat'
                        )
                    decision_match = re.search(r'Decision:(.*)', buy_response)
                    bargain_state_match = re.search(r'Bargain statement:(.*)', buy_response)
                    if decision_match and bargain_state_match:
                        logging.info('First Bargain prompt.\\' + str(buy_actual_prompt) + '\n' + 'First Bargain Response. \\' + buy_response + '\n')
                        decision = decision_match.group(1).lstrip(' ').rstrip('.')
                        bargain_statement = bargain_state_match.group(1).lstrip(' ')
                        if decision.lower() == 'accept':
                            action_text = 'Market' + '.' + action_content + '_' + 'Mat' + '+' + str(buy_req['lowest_price'])
                        else:                          
                            self.bargain_state.update({'in_bargain':True, 'bargain_role':'Buyer', 'bargain_object': seller_id, 'initial_price':buy_req['lowest_price'], 'bargain_memory':[initial_message]})
                            bargain_statement= self.to_Message(bargain_statement, self.agent_id, seller_id)
                            self.bargain_state['bargain_memory'].append(bargain_statement)
                            action_text = 'Market.Bargain_Mat' + '+' + json.dumps(bargain_statement)
                            self.bargain_state['wait_round'] = 0  #第一次buyer发出去的时候的时候应该选择开始计次，不然前面累计，或者每一次send的时候把wait round清零
                            logging.info(str(bargain_statement))
                    else:
                        logging.error('error: Do not find bargain decision in the text. \\' + buy_response)
                        action_text = 'Market' + '.' + action_content + '_' + 'Mat' + '+' + str(buy_req['lowest_price'])
                
            else:
                action_text = 0
        
        action = self.parse_action(action_text)
        self.min_ask_price = self_min_ask_price
        if action == 0:
            self.memory.append('NO-OP')
        else:
            self.memory.append(action.keys())
        return action
    
    def to_Message(self, content = "", speaker_id = "", receiver_id = ""):
        message  = {}
        message["content"] = content
        message["speaker_id"] = speaker_id
        message["receiver_id"] = receiver_id
        return message


    def to_bargain_history(self):
        reverse_roles = {'Buyer':'Seller', "Seller":'Buyer'}
        return "\n".join(
            [
                f"[{reverse_roles[self.bargain_state['bargain_role']]}]: {message['content']}"
                if message['speaker_id'] != self.agent_id
                else f"[{self.bargain_state['bargain_role']}]: {message['content']}"
                for message in self.bargain_state['bargain_memory']
            ]
        )
    
    def Buyer_bargain_response(self, opposite_message, Token_num, initial_price):
        self.bargain_state['wait_round'] = 0
        bargain_prompt = '''You are now playing a role as a game player in a MMO game. You want to use game currency Tokens to buy Material which is important resource in the game from another player and you are bargaining with him.
        Your currently possess {Token_num} Tokens. The Material's initial selling price is {initial_price} Tokens. Your historical chat history with the seller is : [{chat_history}]. You represent the Buyer, and your top price is {top_price} Tokens. 
        Your goal is to buy the Material with a low price not exceeding your inventory, as low as possible. Now, you should only return the natural language response to the seller without any other content or modifiers.
        Here is an example response:
        The price is a little hight, and I can only offer 6 Tokens.'''
        self.bargain_state['bargain_memory'].append(opposite_message)
        bargain_req = {'chat_history': self.to_bargain_history()}
        bargain_req['Token_num'] = Token_num
        bargain_req['top_price']= min(int(initial_price * 0.8), Token_num)
        bargain_req['initial_price'] = initial_price
        bargain_prompt_str = bargain_prompt.format(**bargain_req)

        bargain_response, _, bargain_actual_prompt = llm_server(
                {"prompt": bargain_prompt_str, **self.Prompt_Config}, mode = 'chat'
                )
        
        return_msg =  self.to_Message(content=bargain_response, speaker_id=self.agent_id, receiver_id=opposite_message['speaker_id'])
        logging.info('Buyer Bargain prompt.\\' + str(bargain_actual_prompt) + '\n' + 'Buyer Bargain Rresponse. \\' + bargain_response + '\n' + str(return_msg))
        self.bargain_state['bargain_memory'].append(return_msg)
        return return_msg
    #当obs中包含message时且inbargain = false自动进入seller
    def Seller_bargain_response(self, opposite_message, initial_price):
        self.bargain_state['wait_round'] = 0
        bargain_prompt = '''You are now playing a role as a game player in a MMO game. You want to sell Material which is important resource in the game to other players to obtain game currency Tokens and now a buyer is bargaining with you.
        Your historical chat history with the buyer is : [{chat_history}]. You represent the [Seller], and your bottom price is {bottom_price} Tokens. Your goal is to sell the Material with a satisfactory price, as high as possible. Now, you should only return the natural language response to the [Buyer] without any other content or modifiers.'''
        self.bargain_state['bargain_memory'].append(opposite_message)
        bargain_req = {'chat_history': self.to_bargain_history()}
        bargain_req['bottom_price'] = int(initial_price * 0.8)
        bargain_prompt_str = bargain_prompt.format(**bargain_req)

        bargain_response, _, bargain_actual_prompt = llm_server(
                {"prompt": bargain_prompt_str, **self.Prompt_Config}, mode = 'chat'
                )
        return_msg = self.to_Message(content=bargain_response, speaker_id=self.agent_id, receiver_id=opposite_message['speaker_id'])
        logging.info('Seller Bargain prompt.\\' + str(bargain_actual_prompt) + '\n' + 'Seller Bargain Rresponse. \\' + bargain_response + '\n' + str(return_msg))
        self.bargain_state['bargain_memory'].append(return_msg)
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
        elif component_name == 'Market.Bargain_Mat':
            if action_.isdigit():
                action_num = int(action_) + 1  #deal price
            else:
                #print(action_)
                action_num = json.loads(action_) #response message
        else:
            action_num = int(action_) + 1
        
        return {component_name : action_num}