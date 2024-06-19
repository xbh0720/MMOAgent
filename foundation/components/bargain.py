# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np
from foundation.base.base_component import BaseComponent, component_registry


@component_registry.add
class Bargain(BaseComponent):
    name = "Bargain"
    component_type = None
    required_entities = ["Mat", "Token", "Labor"]
    agent_subclasses = ["BasicPlayer"]
    
    def __init__(
        self,
        *base_component_args,
        bargain_labor=1.0,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        self.bargain_labor=bargain_labor
        assert self.bargain_labor >= 0
        self.chat_history = []
        self.deals = []
        self.messages_step = []

    # Required methods for implementing components
    # --------------------------------------------

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        Add a single action (build) for mobile agents.
        """
        # This component adds 1 action that mobile agents can take: build a house
        if agent_cls_name == "BasicPlayer":
            return 1

        return None

    def get_additional_state_fields(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        For mobile agents, add state fields for building skill.
        """
        return {}

    def component_step(self):
        """
        See base_component.py for detailed description.

        Convert stone+wood to house+coin for agents that choose to build and can.
        """
        world = self.world
        # Apply any building actions taken by the mobile agents
        self.messages_step = []
        self.deals.append([])
        for agent in world.get_random_order_agents():

            message = agent.get_component_action(self.name)
            opposite_messages = [message] if isinstance(message, (int, dict)) else message

            for opposite_message in opposite_messages:
                if opposite_message == 0:  # NO-OP!
                    pass
                elif isinstance(opposite_message, dict): #successful bargain, change selling price change both self.ask_hists and self.asks
                    if 'new_price' in opposite_message.keys():
                        # not typical message
                        new_price = int(opposite_message['new_price'])
                        seller_id = opposite_message['seller_id']
                        buyer_id = opposite_message['buyer_id']
                        buyer = self.world.agents[buyer_id]
                        seller = self.world.agents[seller_id]
                        if self.world.agents[buyer_id].state["inventory"]['Token'] >= new_price:
                            buyer.state["inventory"]['Token'] -= new_price
                            seller.state["inventory"]['Token'] += new_price
                            seller.state["inventory"]['Mat'] -= 1
                            buyer.state["inventory"]['Mat'] += 1
                            self.deals[-1].append(opposite_message)
                        
                    # for in-bargain chat observe
                    else:
                        self.messages_step.append(opposite_message)
                        
                    agent.state["endogenous"]["Labor"] += self.bargain_labor
                else:
                    raise ValueError
        self.chat_history.append(self.messages_step)

    def generate_observations(self):
        """
        See base_component.py for detailed description.

        Here, agents observe their build skill. The planner does not observe anything
        from this component.
        """
        world = self.world
        obs = {a.idx: {'bargain_msg':[]} for a in world.agents}
        for message in self.messages_step:
            receiver_id = message['receiver_id']
            if receiver_id == 'all':
                for a in world.agents:
                    a_id = a.idx
                    if a_id != message['speaker_id']:
                        obs[a_id]['bargain_msg'].append(message)
            else:
                obs[receiver_id]['bargain_msg'].append(message)

        return obs

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.

        Prevent building only if a landmark already occupies the agent's location.
        """

        masks = {}
        # Mobile agents' build action is masked if they cannot build with their
        # current location and/or endowment
        for agent in self.world.agents:
            masks[agent.idx] = np.array([1])

        return masks

    # For non-required customization
    # ------------------------------

    def get_metrics(self):
        """
        Metrics that capture what happened through this component.

        Returns:
            metrics (dict): A dictionary of {"metric_name": metric_value},
                where metric_value is a scalar.
        """

        return {}

    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.

        Re-sample agents' building skills.
        """

        self.chat_history = []
        self.deals = []

    def get_dense_log(self):
        """
        Log builds.

        Returns:
            builds (list): A list of build events. Each entry corresponds to a single
                timestep and contains a description of any builds that occurred on
                that timestep.

        """
        # deal price  of bargain
        return self.deals
