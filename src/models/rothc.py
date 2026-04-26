import numpy as np
import pandas as pd

class RothC:
    """
    Implementation of the Rothamsted Carbon Model (RothC-26.3).
    """
    def __init__(self, clay_pct, p_e=0.75):
        self.clay = clay_pct
        self.p_e = p_e # Potential Evapotranspiration factor
        
        # Default annual decomposition rates (k) in per year
        self.k = {
            'DPM': 10.0,
            'RPM': 0.3,
            'BIO': 0.66,
            'HUM': 0.02,
            'IOM': 0.0
        }
        
        # Distribution of C from decomposed material
        # The proportion of decomposed C that goes to (BIO + HUM) vs CO2
        # is determined by the clay content.
        x = 1.67 * (1.85 + 1.60 * np.exp(-0.078 * self.clay))
        self.f_bio_hum = 1.0 / (x + 1.0)
        
        # Partition between BIO and HUM
        self.f_bio = 0.46
        self.f_hum = 0.54
        
    def get_fT(self, temp):
        """Temperature rate modifier."""
        if temp <= -18.3:
            return 0
        return 47.91 / (1.0 + np.exp(106.06 / (temp + 18.3)))

    def get_fW(self, rainfall, pet, max_acc_def=None):
        """Moisture rate modifier (simplified)."""
        # A more complex implementation would track Soil Moisture Deficit (SMD)
        # For this implementation, we use a simplified rainfall/PET ratio proxy
        # if max_acc_def is not provided.
        if pet == 0: return 1.0
        ratio = rainfall / pet
        if ratio >= 1.0: return 1.0
        return max(0.2, ratio)

    def step(self, pools, temp, rainfall, pet, carbon_input, is_covered=True, dt=1/12):
        """
        Advance one timestep (default 1 month).
        """
        fT = self.get_fT(temp)
        fW = self.get_fW(rainfall, pet)
        fC = 1.0 if is_covered else 0.6
        
        # Combined rate modifier
        rho = fT * fW * fC
        
        new_pools = pools.copy()
        total_decomp = 0
        
        # 1. Decompose existing pools
        decomp = {}
        for pool in ['DPM', 'RPM', 'BIO', 'HUM']:
            # Amount decomposed
            amount = pools[pool] * (1 - np.exp(-self.k[pool] * rho * dt))
            decomp[pool] = amount
            new_pools[pool] -= amount
            total_decomp += amount
            
        # 2. Distribute decomposed C
        # Amount going to BIO + HUM
        to_soil = sum(decomp.values()) * self.f_bio_hum
        new_pools['BIO'] += to_soil * self.f_bio
        new_pools['HUM'] += to_soil * self.f_hum
        
        # 3. Add new carbon input
        # Standard partition for DPM/RPM (e.g., 0.59/0.41 for agricultural crops)
        f_dpm = 0.59
        new_pools['DPM'] += carbon_input * f_dpm * dt
        new_pools['RPM'] += carbon_input * (1 - f_dpm) * dt
        
        return new_pools

def estimate_initial_pools(total_soc):
    """
    Step 4.4: Initial Pool Partitioning using Falloon equation.
    """
    iom = 0.049 * (total_soc ** 1.139)
    active_soc = total_soc - iom
    
    # Initial guesses (will be refined by spin-up)
    return {
        'DPM': active_soc * 0.01,
        'RPM': active_soc * 0.15,
        'BIO': active_soc * 0.03,
        'HUM': active_soc * 0.81,
        'IOM': iom
    }
