import akro
import numpy as np
import garage

class Car():
    def __init__(self, pos=0, speed=0, accel=0):
        self.pos = pos
        self.speed = speed
        self.accel = accel
        self.jerk = 0
    
    def __str__(self):
        return "{:.2f}m - {:.3f}m/s - {:.5f}m/s2".format(self.pos, self.speed, self.accel)

class CarEnv(garage.Environment):
    def __init__(self, specs):
        self.specs = specs
    
    @property
    def observation_space(self):
        return akro.Box(low=np.array([0, self.specs["vMin"], -self.specs["bMax"]]),
                        high=np.array([self.specs["clipdist"], self.specs["vMax"], self.specs["aMax"]]))
    
    @property
    def action_space(self):
        return akro.Box(low=np.array([-1]), high=np.array([1]))

    @property
    def spec(self):
        return garage.EnvSpec(self.observation_space, self.action_space, 1000)
    
    @property
    def render_modes(self):
        return ['ansi']

    def denorm_action(self, action):
        if action[0] < 0:
            return action * self.specs["bMax"]
        return action * self.specs["aMax"]
    
    def encode_observation(self):
        return np.clip(np.array([
            self.othercar.pos - self.owncar.pos,
            self.owncar.speed,
            self.owncar.accel
        ]), self.observation_space.low, self.observation_space.high)
    
    def has_terminated(self):
        return self.othercar.pos <= self.owncar.pos
    
    def calc_reward(self):
        gap = np.abs(self.othercar.pos - self.owncar.pos)
        
        # Wunschabstand reduzieren
        vOpt = min(self.specs["vTarget"], (gap - self.specs["gapMin"])/self.specs["tTarget"])
        rGap = -self.specs["gamma_gap"] * ((vOpt - self.owncar.speed)/self.specs["vTarget"]) ** 2
        
        # Bei Folgefahrt Geschwindigkeitsdifferenz minimieren
        rFollow = 0
        if gap < 200:
            rFollow = -self.specs["gamma_follow"] * ((min(self.othercar.speed, self.specs["vTarget"])-self.owncar.speed)/self.specs["vTarget"]) ** 2
            
        # Beschleunigung minimieren
        rAccel = -self.specs["gamma_accel"] * (self.owncar.accel / self.specs["aMax"])** 2
        
        # Ruck minimieren
        rJerk = -self.specs["gamma_jerk"] * (self.owncar.jerk / self.specs["jComf"]) ** 2
        
        # Kritikalität minimieren
        bKin = (self.owncar.speed - self.othercar.speed) ** 2 / (2*gap+1e-6) * (self.owncar.speed > self.othercar.speed)
        rKrit = -self.specs["gamma_crit"] * (bKin / self.specs["bComf"]) ** 2
        
        r = np.clip(rGap + rFollow + rAccel + rJerk + rKrit, -self.specs["clip_reward"], self.specs["clip_reward"])
        
        info = {
            "rGap": rGap,
            "rFollow": rFollow,
            "rAccel": rAccel,
            "rJerk": rJerk,
            "rKrit": rKrit,
            "vOpt": vOpt,
            "bKin": bKin,
        }
        
        return r, info
    
    def reset(self):
        self.owncar = Car(0, 0, 0)
        self.othercar = Car(2, 0, 0)
        self.otherstate = "accel"
        self.firststep = True
        
        return self.encode_observation(), {}
    
    def move_car(self, car, specs, accel):
        accel = np.clip(accel[0], -specs["bMax"], specs["aMax"])
        
        if accel > car.accel + specs["jMax"] * specs["timestep"]:
            accel = car.accel + specs["jMax"] * specs["timestep"]
        elif accel < car.accel - specs["jMin"] * specs["timestep"]:
            accel = car.accel - specs["jMin"] * specs["timestep"]
        
        jerk = (accel - car.accel) / specs["timestep"]
        
        new_speed = car.speed + accel * specs["timestep"]
        
        if new_speed > specs["vMax"]:
            new_speed = specs["vMax"]
            accel = (new_speed - car.speed) / specs["timestep"]
        elif new_speed < specs["vMin"]:
            new_speed = specs["vMin"]
            accel = (new_speed - car.speed) / specs["timestep"]
        
        car.pos = car.pos + (new_speed + car.speed) / 2 * specs["timestep"]
        car.speed = new_speed
        car.accel = accel
        car.jerk = jerk
        return car
    
    def get_other_act(self):
        if np.random.rand() > 0.8/self.specs["timestep"]:
            commands = ["accel", "hardaccel", "brake", "hardbrake", "fullbrake", "cruise"]
            probs_speeding = [0.1, 0, 0.3, 0.2, 0.1, 0.3]
            probs_running = [0.2, 0.05, 0.2, 0.04, 0.01, 0.5]
            probs_standing = [0.7, 0.1, 0, 0, 0, 0.2]
            p = None
            if self.othercar.speed == self.specs["vMin"]:
                p = probs_standing
            elif self.othercar.speed >= self.specs["vMax"]*.7:
                p = probs_speeding
            else:
                p = probs_running
            
            self.otherstate = np.random.choice(commands, p=p)
        
        if self.otherstate == "accel":
            return [np.random.normal(1, 0.1)]
        elif self.otherstate == "hardaccel":
            return [self.specs["aMax"]]
        elif self.otherstate == "brake":
            return [np.random.normal(-2, 0.1)]
        elif self.otherstate == "hardbrake":
            return [np.random.normal(-5, 2)]
        elif self.otherstate == "fullbrake":
            return [-self.specs["bMax"]]
        else:
            return [np.random.normal(0, 0.005)]
    
    def step(self, action):
        assert not np.isnan(np.sum(action))
        action = self.denorm_action(action)

        action_taken = np.copy(action)
        self.owncar = self.move_car(self.owncar, self.specs, np.copy(action))
        self.othercar = self.move_car(self.othercar, self.specs, self.get_other_act())
        
        obs = self.encode_observation()
        d = self.has_terminated()
        r, i = self.calc_reward()

        for k,v in self.owncar.__dict__.items():
            i["owncar/%s" % k] = v
        for k,v in self.othercar.__dict__.items():
            i["othercar/%s" % k] = v
        i["otherstate"] = self.otherstate

        steptype = garage.StepType.MID
        if self.firststep:
            steptype = garage.StepType.FIRST
            self.firststep = False
        if d:
            steptype = garage.StepType.TERMINAL
                
        return garage.EnvStep(
            self.spec,
            action_taken,
            r,
            obs,
            i,
            steptype
        )    
    def render(self, mode=None):
        s = 'Own car: %s - other car(%s) %s' % (str(self.owncar), self.otherstate, str(self.othercar))
        print(s)
        return s

    def close(self):
        pass
    def visualize(self):
        pass