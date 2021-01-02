from training import ppo_car


def kmh_to_ms(val):
    return val/3.6

specs = {
    "aMax": 2,
    "bMax": 9,
    "bComf": 2,
    "jComf": 2,
    "jMax": 20,
    "jMin": 20,
    "tTarget": 1,
    "gapMin": 2,
    "vTarget": kmh_to_ms(50),
    "vMax": kmh_to_ms(150),
    "vMin": 0,
    "timestep": 1,
    "clipdist": 500,
    "gamma_gap": 0.5,
    "gamma_follow": 0.1,
    "gamma_accel": 0.1,
    "gamma_jerk": 0.1,
    "gamma_crit": 0.1,
    "clip_reward": 10,
}
      
ppo_car(specs=specs)

