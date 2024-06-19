from foundation.agents import agent_registry as agents
from foundation.components import component_registry as components
from foundation.entities import endogenous_registry as endogenous
from foundation.entities import landmark_registry as landmarks
from foundation.entities import resource_registry as resources
from foundation.scenarios import scenario_registry as scenarios


def make_env_instance(scenario_name, **kwargs):
    scenario_class = scenarios.get(scenario_name)
    return scenario_class(**kwargs)
