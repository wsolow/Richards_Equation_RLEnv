"""
Implementation of base model class for use with RE Irrigation

Written by Will Solow, 2025
"""
import numpy as np
from traitlets_pcse import Instance, HasTraits

from richards_model.models.states_rates import ParamTemplate, StatesTemplate, RatesTemplate

class BaseRichardsModel(HasTraits):

    states = Instance(StatesTemplate)
    rates = Instance(RatesTemplate)
    params = Instance(ParamTemplate)

    def __init__(self):
        pass

    def reset(self):
        """
        Reset the model
        """
        raise NotImplementedError

    def calc_rates(self, IRRIG):
        """
        Calculate the rates of change for the state variables
        """
        raise NotImplementedError
        
    def integrate(self):
        """
        Integrate the state variables
        """
        raise NotImplementedError
    
    def run(self, IRRIG, steps:int=1):
        """
        Advances the system state with given number of steps
        """
        steps_done = 0
        while (steps_done < steps):
            steps_done += 1
            self._run(IRRIG)

    def _run(self, IRRIG):
        """
        Make one time step of the simulation.
        """

        self.calc_rates(IRRIG)
        self.integrate()
    
    def get_output(self, ovars:list=None):
        """
        Return the Waterbalance as output
        """
        if ovars is None or ovars == []:
            return np.array([self.states.WBS])
        else:
            output_vars = []
            for i, v in enumerate(ovars):
                if v in self.states.trait_names():
                    output_vars.append(np.array([getattr(self.states, v)]).flatten())
                elif v in self.rates.trait_names():
                    output_vars.append(np.array([getattr(self.rates,v)]).flatten())
    
            return np.hstack(output_vars).flatten()
    
    def get_state_rates(self, var: list=None):
        """
        Return the states and rates
        """
        output_vars = []
        if var is None:
            for s in self.states._find_valid_variables():
                output_vars.append(getattr(self.states, s))
            for r in self.rates._find_valid_variables():
                output_vars.append(getattr(self.rates, r))
        else:
            for v in var:
                if v in self.states._find_valid_variables():
                    output_vars.append(getattr(self.states, v))
                elif v in self.rates._find_valid_variables():
                    output_vars.append(getattr(self.rates, v))
        return output_vars
    
    def set_state_rates(self, vars:list|dict):
        """
        Set all states and rates
        """

        if len(vars) != len(self.states._find_valid_variables()) + len(self.rates._find_valid_variables()):
                raise ValueError("Length of vars does not match states and rates")
        
        if isinstance(vars, dict):
            for k,v in vars.items():
                if k in self.states._find_valid_variables():
                    setattr(self.states, k, v)
                elif k in self.rates._find_valid_variables():
                    setattr(self.rates, k, v)

        elif isinstance(vars, list):
            for i, s in enumerate(self.states._find_valid_variables()):
                setattr(self.states, s, vars[0][i])
            for j, r in enumerate(self.rates._find_valid_variables()):
                setattr(self.rates, r, vars[0][j + len(self.states._find_valid_variables())])

    def get_state_rates_names(self):
        """
        Get names of states and rates
        """
        output_vars = []
        for s in self.states._find_valid_variables():
            output_vars.append(s)
        for r in self.rates._find_valid_variables():
            output_vars.append(r)
        return output_vars
    

        


